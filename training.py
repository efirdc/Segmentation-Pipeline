import os
from collections.abc import Sequence
import torch
import torchio as tio
import time


def transform_label_names(subject, label_map, label_names):
    label_names = label_names.copy()
    for name, params in subject.applied_transforms:
        if params["include"] is not None and label_map["name"] not in params["include"]:
            continue
        if params["exclude"] is not None and label_map["name"] in params["exclude"]:
            continue
        if name == "RemoveLabels":
            removed_labels = params['labels']
            label_names = {name: label for name, label in label_names.items() if label not in removed_labels}
        if name == "RemapLabels":
            remapping = params['remapping']
            for name, label in label_names.items():
                if label in remapping:
                    label_names[name] = remapping[label]
    return label_names


class CudaTimer:
    def __init__(self):
        self.start()

    def start(self):
        self.start_time = time.time()
        self.last_time = self.start_time
        self.timestamps = {}

    def stamp(self, name=None, from_start=False):
        torch.cuda.current_stream().synchronize()
        new_time = time.time()
        if not from_start:
            dt = new_time - self.last_time
        else:
            dt = time.time() - self.start_time
        self.last_time = new_time
        if name:
            self.timestamps[name] = dt
        return dt


class SegmentationTrainer:
    def __init__(
            self,
            input_images: Sequence[str],
            target_label: str,
            save_folder: str,
            sample_rate,
            save_rate,
            val_datasets,
            val_images,
            preload_training_dataset=False
    ):
        self.input_images = input_images
        self.target_label = target_label
        self.save_folder = save_folder
        self.sample_rate = sample_rate
        self.save_rate = save_rate
        self.val_datasets = val_datasets
        self.val_images = val_images
        self.preload_training_dataset = preload_training_dataset

    # Given a list of tio.Subjects, returns (X, y) where X is the input_images, y is the target_label
    # X is a FloatTensor shape (N, C, H, W, D), while y is a LongTensor shape (N, H, W, D)
    def collate_subjects(self, batch: Sequence[tio.Subject]) -> (torch.FloatTensor, torch.LongTensor):
        X = []
        y = []
        for subject in batch:
            images = [subject[image_name].data for image_name in self.input_images]
            X.append(torch.cat(images))
            y.append(subject[self.target_label])
        X = torch.stack(X)
        y = torch.cat(y)
        return X, y

    def train(self, context, iterations, stop_time=None, wandb_logging=False):

        save_folder = f'{self.save_folder}/{context.name}/'
        image_folder = save_folder + "images/"
        for folder in (save_folder, image_folder):
            if not os.path.exists(folder):
                os.makedirs(folder)

        if self.preload_training_dataset:
            print("Preloading training data.")
            context.dataset.preload_subjects()

        def save_context():
            context.save(f"{save_folder}/iter{context.iteration:08}.pt")

        def sample_data(loader):
            while True:
                for batch in loader:
                    yield self.collate_subjects(batch)

        print("Preloading validation data.")
        for val_dataset in self.val_datasets:
            if val_dataset["preload"]:
                val_dataset["X"], val_dataset["y"] = self.collate_subjects(
                    [val_dataset["dataset"][i] for i in range(len(val_dataset["dataset"]))]
                )
        print("Preload finished.")

        loader = sample_data(context.dataloader)

        subject = context.dataset[0]

        target_label_name = context.dataset.target_label
        target_label = subject[target_label_name]
        label_names = target_label["label_names"]
        label_names_transformed = transform_label_names(subject, target_label, label_names)
        label_names_inverse = {name: i for name, i in label_names.items() if name in label_names_transformed}
        label_transforms = []
        for t in subject.history:
            if type(t) not in (tio.RemapLabels, tio.RemoveLabels, tio.SequentialLabels):
                continue
            if t.include and target_label_name not in t.include:
                continue
            if t.exclude and target_label_name in t.exclude:
                continue
            label_transforms.append(t)
        _inverse_label_transform = tio.Compose(label_transforms).inverse(warn=False)
        for inv_t in _inverse_label_transform:
            inv_t.exclude = inv_t.include = None
        print("Inverse label transform", _inverse_label_transform)
        def inverse_label_transform(x):
            return _inverse_label_transform(tio.LabelMap(tensor=x)).data

        timer = CudaTimer()
        for _ in range(iterations):
            timer.start()

            batch = next(loader)
            X, y, subjects = batch
            X = X.to(context.device)
            y = y.to(context.device)

            timer.stamp("Time: Data Loading")

            context.model.train()
            y_pred = context.model(X)
            timer.stamp("Time: Model Forward")
            loss = context.criterion(y_pred, y)
            loss_dict = {"loss": loss.item()}
            if isinstance(loss, tuple):
                loss = loss[0]
                loss_dict = loss[1]
            timer.stamp("Time: Loss Function")

            context.optimizer.zero_grad()
            loss.backward()
            context.optimizer.step()
            context.model.eval()
            timer.stamp("Time: Model Backward")

            y_pred = inverse_label_transform(y_pred.argmax(dim=1))
            y = inverse_label_transform(y)
            timer.stamp("Time: Train Label Inverse")

            train_dice_results = dice_validation(y_pred, y, label_names_inverse, "Training")
            timer.stamp("Time: Train Dice")

            cache = {}
            def predict(subject=None, subjects=None, X=None, y=None):
                if X is None or y is None:
                    if subject:
                        if subject.name in cache:
                            return cache[subject.name]
                        subjects = [subject]
                    X, y, _ = SubjectFolder.collate(subjects)
                X = X.to(context.device)
                y = y.to(context.device)
                with torch.no_grad():
                    y_pred = context.model(X)
                y_pred = inverse_label_transform(y_pred.argmax(dim=1))
                y = inverse_label_transform(y)
                out = (X, y, y_pred)
                for i in range(len(subjects)):
                    cache[subjects[i].name] = (X[i:i+1], y[i:i+1], y_pred[i:i+1])
                return out

            val_dice_results = {}
            for val_dataset in self.val_datasets:
                if context.iteration % val_dataset["interval"] != 0:
                    continue
                if val_dataset["preload"]:
                    _, y, y_pred = predict(
                        subjects=val_dataset["dataset"].subjects, X=val_dataset["X"], y=val_dataset["y"]
                    )
                else:
                    subjects = [val_dataset["dataset"][i] for i in range(len(val_dataset["dataset"]))]
                    _, y, y_pred = predict(subjects=subjects)
                val_dice_results.update(dice_validation(y_pred, y, label_names_inverse, val_dataset["log_prefix"]))
            timer.stamp("Time: Validation Dice")

            val_image_results = {}
            for val_image in self.val_images:
                if context.iteration % val_image["interval"] != 0:
                    continue
                imgs = []
                ys = []
                y_preds = []
                for name in val_image["subjects"]:
                    for val_dset in self.val_datasets:
                        if name in val_dset["dataset"]:
                            dset = val_dset["dataset"]
                    if dset is None:
                        raise ValueError(f"Subject {name} not found in any dataset.")
                    subject = dset[name]
                    img = subject[val_image["image_name"]].data
                    _, y, y_pred = predict(dset[name])
                    img, y, y_pred = [slice_volume(x, val_image["plane"], val_image["slice"]) for x in (img, y, y_pred)]
                    imgs.append(img.unsqueeze(0))
                    ys.append(y.unsqueeze(0))
                    y_preds.append(y_pred.unsqueeze(0))
                img, y, y_pred = [
                    make_grid(x, nrow=val_image["ncol"], pad_value=-1, padding=1)[0].cpu()
                    for x in (imgs, ys, y_preds)
                ]
                H, W = img.shape
                fig = plt.figure(figsize=tuple(np.array((W, H)) / 7.))
                plt.imshow(img, cmap="gray", vmin=-1, vmax=1)
                X_grid, Y_grid = np.meshgrid(np.linspace(0, W - 1, W), np.linspace(0, H - 1, H))
                options = dict(linewidths=1.5, alpha=1.)
                warnings.filterwarnings("ignore")
                cmap = [None, "r", "g", "b", "y", "c", "m"] \
                       + list(get_cmap("Accent").colors) + list(get_cmap("Dark2").colors) \
                       + list(get_cmap("Set1").colors) + list(get_cmap("Set2").colors) \
                       + list(get_cmap("tab20").colors)
                contours = []
                for name, label_id in label_names_inverse.items():
                    contour = plt.contour(X_grid, Y_grid, y == label_id, levels=[0.5], colors=cmap[label_id:label_id+1],
                                          **options)
                    plt.contour(X_grid, Y_grid, y_pred == label_id, levels=[0.95], linestyles="dashed",
                                colors=cmap[label_id:label_id+1], **options)
                    contours.append(contour)
                if val_image["legend"]:
                    plt.legend([contour.legend_elements()[0][0] for contour in contours], label_names_inverse.items(),
                               ncol=3, bbox_to_anchor=(0.5, 0), loc='upper center', fancybox=True)
                warnings.resetwarnings()
                plt.tick_params(which='both', bottom=False, top=False, left=False, labelbottom=False, labelleft=False)
                buf = io.BytesIO()
                fig.savefig(buf, bbox_inches="tight", pad_inches=0.0, facecolor="black")
                buf.seek(0)
                pil_image = Image.open(buf)
                plt.close(fig)
                val_image_results[val_image["log_name"]] = wandb.Image(pil_image)
            cache = {}
            timer.stamp("Time: Validation Images")

            if context.iteration != 0 and context.iteration % self.save_rate == 0:
                save_context()
            timer.stamp("Time: Saving Context")
            timer.stamp("Time: Total", from_start=True)

            wandb.log({**loss_dict, **train_dice_results, **val_dice_results, **val_image_results, **timer.timestamps})

            context.iteration += 1

            if EXIT.is_set() or (stop_time is not None and time.time() > stop_time):
                if EXIT.is_set():
                    print("Training stopped early due to manual exit signal.")
                else:
                    print("Training time expried.")
                break

        print("Saving context...")
        save_context()
from abc import ABC, abstractmethod
from typing import Sequence, Dict, Union, Any

import torchio as tio

from utils import as_set, is_sequence, vargs_or_sequence


class SubjectFilter(ABC):
    """ Abstract class for filtering subjects

    All subclasses must overwrite the `subject_filter` method,
    which takes a `tio.Subject` a bool that is `True` if the subject
    will be kept and `False` if the subject will be filtered.

    """
    def __call__(
            self,
            *subjects: Union[tio.Subject, Sequence[tio.Subject]]
    ):
        subjects = vargs_or_sequence(subjects)
        if is_sequence(subjects) and all(isinstance(subject, tio.Subject) for subject in subjects):
            return list(filter(self.subject_filter, subjects))
        else:
            raise ValueError("A SubjectFilter can only be applied to a sequence of tio.Subject, "
                             f"not {subjects}")

    @abstractmethod
    def subject_filter(self, subject: tio.Subject) -> bool:
        raise NotImplementedError()

    def __and__(self, other):
        assert other is SubjectFilter
        return ComposeFilters(self, other)

    def __or__(self, other):
        assert other is SubjectFilter
        return AnyFilter(self, other)

    def __sub__(self, other):
        assert other is SubjectFilter
        return ComposeFilters(self, NegateFilter(other))

    def __neg__(self):
        return NegateFilter(self)

    def __invert__(self):
        return NegateFilter(self)


class RequireAttributes(SubjectFilter):
    """ Filters subjects which do not have the required attributes

    Args:
        attributes: Defines the required attributes for subjects. It can be one of:

            - A sequence of keys that are required to be in the subject dictionary.

            - A dictionary that defines both required keys and values in the subject dictionary.

    Examples:
        >>> # Requires that subjects have T1w, T2w images and anatomical labels
        >>> RequireAttributes(['t1', 't2', 'anatomical_label']),
        >>>
        >>> # Requires that subjects are between 10 and 34 years old and have no disease
        >>> RequireAttributes({'age': range(10, 35), "disease": False})
    """
    def __init__(
            self,
            attributes: Union[Sequence[str], Dict[str, Any]]
    ):
        self.attributes = attributes

    def subject_filter(self, subject):
        if isinstance(self.attributes, list):
            return all(attribute in subject for attribute in self.attributes)
        if isinstance(self.attributes, dict):
            if any(attribute not in subject for attribute in self.attributes.keys()):
                return False
            return all(
                not as_set(attrib_value).isdisjoint(as_set(subject.get(attrib_name)))
                for attrib_name, attrib_value in self.attributes.items()
            )


class ForbidAttributes(SubjectFilter):
    """ Filters subjects which have forbidden attributes

    Args:
        attributes: Defines the forbidden attributes for subjects. It can be one of:

            - A sequence of keys that are forbidden to be in the subject dictionary.

            - A dictionary that defines forbidden values for keys in the subject dictionary.
              Note that the keys themselves are not forbidden. The constraint is on the values only.

    Examples:
        >>> # Forbid subjects that have an anatomical label
        >>> ForbidAttributes(['anatomical_label']),
        >>>
        >>> # Forbid subjects that are in the validation set
        >>> validation_subjects = ["subject_05", "subject_08"]
        >>> ForbidAttributes({'name': validation_subjects})
    """
    def __init__(
            self,
            attributes: Union[Sequence[str], Dict[str, Any]]
    ):
        self.attributes = attributes

    def subject_filter(self, subject):
        if isinstance(self.attributes, list):
            return not any(attribute in subject for attribute in self.attributes)
        if isinstance(self.attributes, dict):
            attributes_in_subject = {
                attrib_name: attrib_value for attrib_name, attrib_value in self.attributes.items()
                if attrib_name in subject
            }
            return all(
                as_set(attrib_value).isdisjoint(as_set(subject.get(attrib_name)))
                for attrib_name, attrib_value in attributes_in_subject.items()
            )


class ComposeFilters(SubjectFilter):
    """ Combines `SubjectFilter`s using a logical and.

    Args:
        filters: A sequence of filters. If any filter returns False the subject will be filtered.
    """
    def __init__(self, *filters: Union[SubjectFilter, Sequence[SubjectFilter]]):
        self.filters = vargs_or_sequence(filters)

    def subject_filter(self, subject):
        return all(f.subject_filter(subject) for f in self.filters)


class AnyFilter(SubjectFilter):
    """ Combines `SubjectFilter`s using a logical or.

    Args:
        filters: A sequence of filters. If any filter returns True the subject will not be filtered.
    """
    def __init__(self, *filters: Union[SubjectFilter, Sequence[SubjectFilter]]):
        self.filters = vargs_or_sequence(filters)

    def subject_filter(self, subject):
        return any(f.subject_filter(subject) for f in self.filters)


class NegateFilter(SubjectFilter):
    """ Negates the provided `SubjectFilter`.

    Args:
        filter: A filter to be negated.
    """
    def __init__(self, filter: SubjectFilter):
        self.filter = filter

    def subject_filter(self, subject):
        return not self.filter.subject_filter(subject)


from typing import TypeVar, Generic, Optional as __optional__
import numpy.typing
from Sofa.Core import Node, Object, LinkPath
import numpy
from numpy.typing import ArrayLike

T = TypeVar("T", bound=object)

# This is a generic type 'T' implemented without PEP 695 (as it needs python 3.12)
class Data(Generic[T]):
    linkpath: LinkPath
    value: T

Optional = __optional__
SofaArray = numpy.ndarray | list

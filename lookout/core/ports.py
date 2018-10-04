import sys

# Fix `typing.Type` different behaviour in python3.5 & python3.6+
if sys.version_info[:2] == (3, 5):
    from typing import Generic, TypeVar

    # Internal type variable used for Type[].
    CT_co = TypeVar("CT_co", covariant=True, bound=type)

    # This is not a real generic class.  Don't use outside annotations.
    class Type(Generic[CT_co], extra=type):
        """A special construct usable to annotate class objects.
        For example, suppose we have the following classes::
          class User: ...  # Abstract base for User classes
          class BasicUser(User): ...
          class ProUser(User): ...
          class TeamUser(User): ...
        And a function that takes a class argument that's a subclass of
        User and returns an instance of the corresponding class::
          U = TypeVar('U', bound=User)
          def new_user(user_class: Type[U]) -> U:
              user = user_class()
              # (Here we could write the user object to a database)
              return user
          joe = new_user(BasicUser)
        At this point the type checker knows that joe has type BasicUser.
        """

        __slots__ = ()
else:
    from typing import Type  # noqa: F401

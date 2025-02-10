class MyClass:
    """A silly class to demonstrate Python's functional interface"""

    def __init__(self, thing1, thing2):
        """A constructor requiring two arguments"""
        self.thing1 = thing1
        self.thing2 = thing2

    def __call__(self, *args, **kwargs):
        """A generic call function"""
        print("I know about:", self.thing1, self.thing2)
        print("arguments   :", args)
        print("keywords    :", kwargs)


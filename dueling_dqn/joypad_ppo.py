
from nes_py.wrappers.joypad_space import JoypadSpace

class JoyPadSpacePPO(JoypadSpace):
    def reset(self, *args, **kwargs):
        # Remove 'seed' from kwargs if it exists
        kwargs.pop('seed', None)
        # Call the original reset method
        return super().reset()
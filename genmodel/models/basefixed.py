from pylego.model import Model


class BaseFixed(Model):

    def __init__(self, model, flags, *args, **kwargs):
        self.flags = flags
        super().__init__(model=model, *args, **kwargs)

    def prepare_batch(self, data):
        batch = super().prepare_batch(data)
        return (b.double() for b in batch)

from pylego.model import Model


class BaseFixed(Model):

    def __init__(self, model, flags, *args, **kwargs):
        self.flags = flags
        super().__init__(model=model, *args, **kwargs)

    def prepare_batch(self, data):
        batch = super().prepare_batch(data)
        if isinstance(batch, tuple):
            return tuple(b.double() for b in batch)
        else:
            return batch.double()

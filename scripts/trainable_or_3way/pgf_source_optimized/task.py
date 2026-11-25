"""Artifact task retains physical batch, labels, and forward arithmetic."""
import torch
from scripts.trainable_or_3way.pgf_source.artifact_training import Task as OriginalTask
from scripts.trainable_or_3way.pgf_source.common import stamp, require
from .frames import TaskInputs


class Task(OriginalTask):
    def __init__(self, *args, input_config=None, **kwargs):
        super().__init__(*args, **kwargs)
        self.inputs = (TaskInputs(self.records, self.stamps, input_config)
                       if self.mod == "video" and self.kind == "artifact" and input_config else None)

    def begin_inputs(self, order):
        if self.inputs: self.inputs.begin(order)

    def begin_group(self, indices):
        if self.inputs: self.inputs.begin_group(indices)

    def close_inputs(self):
        if self.inputs: self.inputs.close()

    def forward(self, model, indices):
        if self.inputs is None: return super().forward(model, indices)
        for ix in indices:
            r = self.records[ix]
            require(stamp(r) == self.stamps[r.sample_id], "Training media changed: "+r.sample_id)
        values = self.inputs.tensor(indices, self.device)
        return model(values).logits, torch.as_tensor(self.y[indices], device=self.device)

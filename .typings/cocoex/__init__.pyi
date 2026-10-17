"""The part of cocoex (coco-experiment) that optbench's tests use."""

class Suite:
    def __init__(
        self,
        suite_name: str,
        suite_instance: str,
        suite_options: str,
    ) -> None: ...

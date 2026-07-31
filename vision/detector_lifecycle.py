class DetectorRegistry:
    """Load every configured detector once and select which ones run."""

    def __init__(self, profile, *, fx=None, fy=None, cx=None, cy=None):
        self.profile = profile
        self.detector_names = self._validated_detector_names(profile)
        self.detectors = {}
        self.active_names = self.detector_names
        self.current_task = None

        for name in self.detector_names:
            spec = profile.DETECTOR_SPECS[name]
            try:
                self.detectors[name] = spec["class"](
                    **self._detector_kwargs(
                        profile,
                        spec,
                        fx=fx,
                        fy=fy,
                        cx=cx,
                        cy=cy,
                    )
                )
            except Exception as exc:
                model_path = spec.get("model_path")
                model_detail = (
                    ""
                    if model_path is None
                    else f" (model_path={model_path})"
                )
                raise RuntimeError(
                    f"Failed to load detector '{name}'{model_detail}: {exc}"
                ) from exc

    @staticmethod
    def _validated_detector_names(profile):
        names = tuple(profile.STARTUP_DETECTORS)
        spec_names = tuple(profile.DETECTOR_SPECS)

        if not names:
            raise ValueError("STARTUP_DETECTORS must not be empty")
        if len(names) != len(set(names)):
            raise ValueError("STARTUP_DETECTORS contains duplicate detector names")

        unknown = [name for name in names if name not in profile.DETECTOR_SPECS]
        missing = [name for name in spec_names if name not in names]
        if unknown or missing:
            raise ValueError(
                "STARTUP_DETECTORS must list every DETECTOR_SPECS entry exactly "
                f"once; unknown={unknown}, missing={missing}"
            )

        for task, wanted in profile.TASK_DETECTOR_MAP.items():
            invalid = sorted(set(wanted) - set(names))
            if invalid:
                raise ValueError(
                    f"TASK_DETECTOR_MAP[{task!r}] references unknown detectors: "
                    f"{invalid}"
                )

        return names

    @staticmethod
    def _detector_kwargs(profile, spec, *, fx=None, fy=None, cx=None, cy=None):
        detector_kwargs = {
            "fx": fx,
            "cx": cx,
            "camera_width": profile.CAMERA_WIDTH,
        }
        if "model_path" in spec:
            detector_kwargs.update({
                "model_path": spec["model_path"],
                "device": profile.DEVICE,
                "tolerance_ratio": profile.TOLERANCE_RATIO,
                "tolerance_deg": profile.TOLERANCE_DEG,
            })
        if spec.get("uses_full_intrinsics"):
            detector_kwargs.update({
                "fy": fy,
                "cy": cy,
            })
        detector_kwargs.update(spec.get("kwargs", {}))
        return detector_kwargs

    def select_task(self, task):
        """Select already-loaded detectors for a task.

        Returns ``False`` for an unknown task without changing the current
        selection. Repeated valid selections are intentionally idempotent.
        """

        wanted = self.profile.TASK_DETECTOR_MAP.get(task)
        if wanted is None:
            return False

        self.current_task = task
        self.active_names = tuple(
            name for name in self.detector_names if name in wanted
        )
        return True

    def active_items(self):
        for name in self.active_names:
            yield name, self.detectors[name]

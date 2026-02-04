# microblocks/registry.py
import importlib, inspect, pkgutil, pathlib
from onnx import TensorProto

class MappingHandle:
    """
    Wraps a stage mapping so you can call .getParam("coeff")
    and always resolve to the correct top-level output name.
    Prefers suffixed '.function' variants to avoid SSA collisions.
    """

    def __init__(self, stage_name: str, class_name: str, outputs: dict):
        self.stage_name = stage_name
        self.class_name = class_name
        self.outputs = outputs  # dict of {name: resolved_name}

    def getParam(self, coeff_name_val: str) -> str:
        """
        Resolve a coefficient/output name to the correct top-level variant.
        Resolution order:
          1. Exact suffixed match (foo.function)
          2. Raw key + ".function" if available
          3. Stage-scoped suffixed (stage.foo.function)
          4. Stage-scoped raw (stage.foo)
          5. Raw key (foo) only if no suffixed variant exists
        """
        # 1. Exact suffixed match
        if coeff_name_val.endswith(".function") and coeff_name_val in self.outputs:
            return self.outputs[coeff_name_val]["name"]

        # 2. Raw key + ".function"
        suffixed = coeff_name_val + ".function"
        if suffixed in self.outputs:
            return self.outputs[suffixed]["name"]

        # 3. Stage-scoped suffixed
        stage_key_suffixed = f"{self.stage_name}.{coeff_name_val}.function"
        if stage_key_suffixed in self.outputs:
            return self.outputs[stage_key_suffixed]["name"]

        # 4. Stage-scoped raw
        stage_key = f"{self.stage_name}.{coeff_name_val}"
        if stage_key in self.outputs:
            return self.outputs[stage_key]["name"]

        # 5. Raw fallback
        if coeff_name_val in self.outputs:
            return self.outputs[coeff_name_val]["name"]

        raise KeyError(
            f"Coeff '{coeff_name_val}' not found in stage '{self.stage_name}' "
            f"(class {self.class_name})"
        )

    def listOutputs(self) -> list[str]:
        """
        Return a list of all available output tensor names for this stage.
        Useful for debugging or inspection.
        """
        return list(self.outputs.keys())

class Registry:
    """
    Singleton registry for microblocks and dynamic outputs.
    Provides fluent API: Registry.getInstance().getMapping(...).getParam(...)
    """
    _instance = None

    def __new__(cls, *args, **kwargs):
        if cls._instance is None:
            cls._instance = super().__new__(cls)
            cls._instance._registry = {}
            cls._instance._outputs_map = {}
            cls._instance._dynamic_map = {}
        return cls._instance

    @classmethod
    def getInstance(cls):
        """Return the singleton instance."""
        return cls()

    def import_all_microblocks(self, base="microblocks"):
        """
        Recursively import all modules under ./microblocks and auto‑register
        any class that declares `name` and `version`.
        """
        base_path = pathlib.Path(__file__).parent
        for finder, modname, ispkg in pkgutil.walk_packages([str(base_path)], prefix=f"{base}."):
            try:
                module = importlib.import_module(modname)
            except Exception as e:
                print(f"[WARN] Failed to import {modname}: {e}")
                continue

            for _, obj in inspect.getmembers(module, inspect.isclass):
                if hasattr(obj, "name") and hasattr(obj, "version"):
                    if obj.__name__ == "MicroblockBase":
                        continue
                    key = (obj.name, obj.version)
                    self._registry[key] = obj
                    print(f"[DEBUG] Registered {key} → {obj.__name__}")

    def dump_registry(self):
        return self._registry

    def clear_all_outputs(self):
        self._outputs_map.clear()

    def register_outputs(self, stage_name: str, class_name: str, outputs: dict):
        # Check for duplicate output names across all stages
        for out_name in outputs.values():
            for existing_stage, spec in self._outputs_map.items():
                if out_name in spec["outputs"].values():
                    raise RuntimeError(
                        f"SSA violation: output '{out_name}' already registered "
                        f"by stage '{existing_stage}' (class {spec['class_name']}). "
                        f"Cannot re-register in stage '{stage_name}' (class {class_name})."
                    )
        # If no collision, register normally
        self._outputs_map[stage_name] = {"class_name": class_name, "outputs": outputs}

    def set_dynamic_map(self, dynamic_map: dict):
        self._dynamic_map = dynamic_map

    def getMapping(self, family_name: str, prev_stages: list) -> MappingHandle:
        """
        Resolve a stage by family name from prev_stages.
        Returns a MappingHandle with .getParam().
        """

        # First pass: try to match family name directly from dynamic_map
        for stage_name in prev_stages:
            if stage_name not in self._dynamic_map:
                continue

            #print(self._dynamic_map)
            spec = self._dynamic_map[stage_name]
            fam = spec.get("family") or spec.get("class")
            print(fam)
            print(family_name)
            if fam == family_name:
                class_name = self._outputs_map[stage_name]["class_name"]
                outputs = self._outputs_map[stage_name]["outputs"]
                return MappingHandle(stage_name, class_name, outputs)

        # Second pass: deduce by class_name in outputs_map
        for stage_name, spec in self._outputs_map.items():
            if spec["class_name"] == family_name:
                return MappingHandle(stage_name, spec["class_name"], spec["outputs"])

        raise KeyError(f"Family '{family_name}' not found in prev_stages or outputs_map")

    # NEW: helper to check if a tensor name has already been produced
    def has_output(self, name: str) -> bool:
        for stage_spec in self._outputs_map.values():
            outputs = stage_spec["outputs"]
            if name in outputs.values():
                return True
        return False

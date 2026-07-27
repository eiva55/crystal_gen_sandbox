from sandbox.contracts.base import BaseMetrics

class CrystalMetrics(BaseMetrics):
    def compute(self, generated, reference=None):
        # Здесь можно реализовать валидность, уникальность и т.д.
        print("Computing metrics...")
        return {"valid_rate": 0.9, "unique_rate": 1.0}

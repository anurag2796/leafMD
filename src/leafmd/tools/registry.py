import json
import shutil
import logging
from pathlib import Path
from datetime import datetime
from typing import Dict, List, Optional

logger = logging.getLogger(__name__)


def _version_sort_key(version_str: str):
    """
    Parse a version string like '1.10.2' into a tuple of ints for proper sorting.

    Falls back to string comparison if parsing fails.
    """
    try:
        return tuple(int(x) for x in version_str.split("."))
    except (ValueError, AttributeError):
        return (0, 0, 0)


class ModelRegistry:
    """Manages model versions and deployment status."""

    def __init__(self, registry_dir: str = "./models"):
        self.registry_dir = Path(registry_dir)
        self.registry_dir.mkdir(exist_ok=True, parents=True)
        self.index_file = self.registry_dir / "registry.json"
        self._load_index()

    def _load_index(self):
        """Load registry index."""
        if self.index_file.exists():
            with open(self.index_file) as f:
                self.index = json.load(f)
        else:
            self.index = {
                "versions": {},
                "production": None,
                "staging": None,
            }

    def _save_index(self):
        """Save registry index."""
        with open(self.index_file, "w") as f:
            json.dump(self.index, f, indent=2)

    def register(
        self,
        model_path: Path,
        version: str,
        metrics: Optional[Dict] = None,
        metadata: Optional[Dict] = None,
        force: bool = False,
    ) -> bool:
        """
        Register a new model version.

        Args:
            model_path: Path to model .pt file.
            version:    Semantic version (e.g., '1.0.0').
            metrics:    Performance metrics dictionary.
            metadata:   Additional metadata.
            force:      Overwrite existing version without prompting.

        Returns:
            True if registration succeeded.
        """
        if not model_path.exists():
            logger.error(f"❌ Model not found: {model_path}")
            return False

        if version in self.index["versions"]:
            if not force:
                logger.warning(
                    f"⚠️  Version {version} already exists. "
                    "Use force=True (or --force) to overwrite."
                )
                return False
            logger.info(f"🔄 Overwriting existing version {version}")

        # Create version directory
        version_dir = self.registry_dir / f"v{version}"
        version_dir.mkdir(exist_ok=True, parents=True)

        # Copy model
        model_dest = version_dir / "best.pt"
        shutil.copy2(model_path, model_dest)

        # Load metrics if not provided
        if metrics is None:
            metrics_file = model_path.parent.parent / "metrics.json"
            if metrics_file.exists():
                with open(metrics_file) as f:
                    metrics = json.load(f)
            else:
                metrics = {}

        # Get model size
        size_mb = model_dest.stat().st_size / (1024 * 1024)

        # Create version entry
        version_info = {
            "version": version,
            "path": str(model_dest),
            "registered_at": datetime.now().isoformat(),
            "status": "testing",
            "metrics": metrics,
            "size_mb": round(size_mb, 2),
            "metadata": metadata or {},
        }

        self.index["versions"][version] = version_info
        self._save_index()

        logger.info(f"✅ Registered model v{version}")
        logger.info(f"   Path: {model_dest}")
        logger.info(f"   Size: {size_mb:.2f} MB")
        if metrics:
            logger.info(f"   mAP@50: {metrics.get('map50', 'N/A')}")

        return True

    def list_versions(self, status: Optional[str] = None) -> List[Dict]:
        """
        List all registered versions.

        Args:
            status: Filter by status (testing/staging/production).

        Returns:
            List of version dictionaries, sorted by semantic version (descending).
        """
        versions = list(self.index["versions"].values())

        if status:
            versions = [v for v in versions if v["status"] == status]

        # Semantic version sort (handles 1.10.0 > 1.9.0 correctly)
        versions.sort(
            key=lambda x: _version_sort_key(x["version"]),
            reverse=True,
        )

        return versions

    def compare(self, version1: str, version2: str):
        """Compare two model versions."""
        v1 = self.index["versions"].get(version1)
        v2 = self.index["versions"].get(version2)

        if not v1:
            logger.error(f"❌ Version {version1} not found")
            return
        if not v2:
            logger.error(f"❌ Version {version2} not found")
            return

        print(f"\n{'=' * 70}")
        print(f"COMPARISON: v{version1} vs v{version2}")
        print(f"{'=' * 70}\n")

        # Metrics comparison
        metrics_keys = set(v1["metrics"].keys()) | set(v2["metrics"].keys())

        print("📊 Metrics:")
        print(f"{'Metric':<20} | v{version1:<10} | v{version2:<10} | Diff")
        print("-" * 70)

        for key in sorted(metrics_keys):
            val1 = v1["metrics"].get(key, 0)
            val2 = v2["metrics"].get(key, 0)

            if isinstance(val1, (int, float)) and isinstance(val2, (int, float)):
                diff = val2 - val1
                diff_pct = (diff / val1 * 100) if val1 != 0 else 0
                diff_str = f"{diff:+.4f} ({diff_pct:+.1f}%)"

                if diff > 0:
                    symbol = "🟢"
                elif diff < 0:
                    symbol = "🔴"
                else:
                    symbol = "⚪"

                print(
                    f"{key:<20} | {val1:<10.4f} | {val2:<10.4f} | "
                    f"{symbol} {diff_str}"
                )
            else:
                print(f"{key:<20} | {val1:<10} | {val2:<10} | -")

        # Size comparison
        print(f"\n📦 Model Size:")
        print(f"   v{version1}: {v1['size_mb']:.2f} MB")
        print(f"   v{version2}: {v2['size_mb']:.2f} MB")
        print(f"   Difference: {v2['size_mb'] - v1['size_mb']:+.2f} MB")

        # Status
        print(f"\n🔖 Status:")
        print(f"   v{version1}: {v1['status']}")
        print(f"   v{version2}: {v2['status']}")

    def promote(self, version: str, to: str = "production") -> bool:
        """Promote a version to a deployment stage."""
        if version not in self.index["versions"]:
            logger.error(f"❌ Version {version} not found")
            return False

        valid_stages = ["testing", "staging", "production"]
        if to not in valid_stages:
            logger.error(f"❌ Invalid stage: {to}. Choose from: {valid_stages}")
            return False

        # Update version status
        self.index["versions"][version]["status"] = to

        # Update stage pointer
        if to in ["staging", "production"]:
            self.index[to] = version

        self._save_index()

        logger.info(f"✅ Promoted v{version} to {to}")

        if to == "production":
            logger.info(f"\n⚠️  PRODUCTION DEPLOYMENT")
            logger.info(
                f"   Update mobile app to use: models/v{version}/best.mlpackage"
            )

        return True

    def rollback(self, version: str) -> bool:
        """Rollback production to a previous version."""
        if version not in self.index["versions"]:
            logger.error(f"❌ Version {version} not found")
            return False

        current_prod = self.index.get("production")

        if current_prod == version:
            logger.warning(f"⚠️  v{version} is already in production")
            return False

        logger.info(f"🔄 Rolling back from v{current_prod} to v{version}")
        return self.promote(version, "production")

    def print_table(self, versions: List[Dict]):
        """Print versions as formatted table."""
        if not versions:
            print("No versions found")
            return

        print(f"\n{'=' * 90}")
        print("MODEL REGISTRY")
        print(f"{'=' * 90}\n")

        print(
            f"{'Version':<12} | {'mAP@50':<8} | {'Size':<8} | "
            f"{'Status':<12} | {'Date':<12}"
        )
        print("-" * 90)

        for v in versions:
            version = v["version"]
            map50 = v["metrics"].get("map50", 0)
            size = f"{v['size_mb']:.1f}MB"
            status = v["status"]
            date = v["registered_at"][:10]

            # Add emoji for deployment stages
            if status == "production":
                status_display = f"🌟 {status}"
            elif status == "staging":
                status_display = f"🚀 {status}"
            else:
                status_display = status

            print(
                f"v{version:<11} | {map50:<8.4f} | {size:<8} | "
                f"{status_display:<12} | {date:<12}"
            )

        print()
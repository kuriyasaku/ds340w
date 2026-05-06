from src.config.paths import ensure_paths
from src.data.wsi_preprocess import WSIPreprocessor
from src.features.phikon_wrapper import FeatureExtractor
from src.infer.rank_export import RankExporter
from src.audit.audit_runner import AuditRunner
from src.train.train_mil import MILTrainer
from src.utils.seed_utils import seed_everything


def main():
    ensure_paths()
    seed_everything(42)
    WSIPreprocessor().run()
    FeatureExtractor().run()
    MILTrainer().fit()
    RankExporter().run()
    AuditRunner().run_all()


if __name__ == "__main__":
    main()

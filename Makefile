.PHONY: test discovery e01 e02 e03 e04 e06 e07 e09 e10 e11 all-discovery

test:
	PYTHONPATH=src python3 -m pytest tests/ -v

e01:
	PYTHONPATH=src python3 experiments/E01_ground_truth/run.py --stage discovery
	PYTHONPATH=src python3 experiments/E01_ground_truth/run.py --stage confirmation

e02:
	PYTHONPATH=src python3 experiments/E02_identification/run.py

e03:
	PYTHONPATH=src python3 experiments/E03_hidden_vulnerability/run.py --stage discovery

e04:
	PYTHONPATH=src python3 experiments/E05_information_matched/run.py

e06:
	PYTHONPATH=src python3 experiments/E06_contrast_robustness/run.py

e07:
	PYTHONPATH=src python3 experiments/E07_future_vulnerability/run.py

e09:
	PYTHONPATH=src python3 experiments/E09_baselines/run.py

e10:
	PYTHONPATH=src python3 experiments/E10_ablations/run.py

e11:
	PYTHONPATH=src python3 experiments/E11_negative_controls/run.py

all-discovery: test e01 e02 e03 e04 e06 e07 e09 e10 e11
	@echo "All RUN-status experiments completed. E08 (SAC) and standard-benchmark validation remain NOT_RUN -- see docs/DEVIATIONS.md."

"""Use the original detector predictor with follow-up runtime isolation."""
import sys

from common import PREVIOUS, initialize_runtime

initialize_runtime()
sys.path.append(str(PREVIOUS))

from predict_detector import main


if __name__ == "__main__":
    main()

"""Use the original historical localization scorer with follow-up runtime isolation."""
import sys

from common import PREVIOUS, initialize_runtime

initialize_runtime()
sys.path.append(str(PREVIOUS))

from score_detector import main


if __name__ == "__main__":
    main()

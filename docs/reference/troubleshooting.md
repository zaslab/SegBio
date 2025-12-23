# Troubleshooting

## Napari / Qt issues
- If the GUI does not open, verify a Qt backend is installed (`pyqt5` or `pyside2`).
- Conflicting Qt bindings can cause `QtBindingsNotFoundError`.

## Common segmentation issues
- Worms touching/overlapping: adjust thresholds and watershed settings.
- Over-splitting: reduce watershed aggressiveness or smooth distance transform.
- Under-splitting: increase separation cues (seed/boundary quality) or tune thresholds.

## Reproducibility
- Log: model weights, commit hash, training args, dataset generator version.

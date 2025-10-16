- python 3.12 will throw `Assertion failed: (range.location + range.length <= dataLength), function __CFDataValidateRange, file CFData.c` when `import keyboard`
- need update [launch.spec](launch.spec) if not using python 3.12 or use .venv instead of venv
.
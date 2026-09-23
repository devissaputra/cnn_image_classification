# Ethics and Limits

This project uses a public handwritten-digit benchmark and does not make consequential decisions about people.

The main caution is generalization. The images are small, clean, and standardized. A CNN that performs well here may behave very differently on handwriting collected from phones, scanned forms, or real documents.

For practical use, I would test robustness on the actual image source and inspect class-specific errors before relying on the model.

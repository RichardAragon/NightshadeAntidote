# Nightshade Antidote

## Overview

Nightshade Antidote is an image forensics tool used to analyze digital images for signs of manipulation or forgery. It implements several common techniques used in image forensics including:

- Metadata analysis
- Copy-move forgery detection 
- Frequency domain analysis
- JPEG compression artifacts analysis

The tool takes an input image, performs analysis using the above techniques, and outputs a report summarizing the findings.

## Requirements

Nightshade Antidote requires the following Python packages:

- OpenCV
- Numpy 
- Matplotlib
- Scipy
- PIL
- Collections
- Scikit-learn
- Exiftool

## Usage

To use Nightshade Antidote, simply run the Python script on an input image:

```
python nightshade_antidote.py input.jpg
```

This will perform forensics analysis on `input.jpg` and output the results to the console and generate plots where relevant.

The script contains several functions that can be called independently to perform specific analyses:

- `detect_copy_move` - Detect copy-move forgery
- `analyze_metadata` - Extract and print metadata
- `spectral_analysis` - Frequency domain analysis
- `pixel_ordering_check` - Check DCT coefficients
- `compression_artifacts_check` - Check for JPEG artifacts
- `file_format_check` - Verify file format
- `output_report` - Generate analysis report

## Output

Nightshade Antidote will output a comprehensive analysis report for the input image including:

- Metadata summary
- Copy-move forgery detection results
- Frequency domain analysis and plots
- JPEG compression artifacts analysis
- File format verification

Any anomalies or indications of manipulation will be highlighted in the report.

## Credits

Nightshade Antidote was created by Richard Aragon. The code implements common digital image forensics techniques based on research papers and books in the field.

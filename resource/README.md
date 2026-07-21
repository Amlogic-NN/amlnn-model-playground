# Quantization Dataset Preparation

## Dataset Availability and Licensing

This repository includes only datasets or calibration samples that the project maintainers are permitted to redistribute.

Some supported models use third-party datasets that are **not included** in this repository because their licenses, terms of use, or redistribution conditions do not permit us to distribute copies directly. References to these datasets are provided only to explain how the models may be calibrated or evaluated.

When a third-party dataset is included:

* It remains subject to its original license and copyright terms.
* The Apache License 2.0 covering this repository does not replace or override the dataset's original license.
* The applicable third-party license and attribution notice must be retained.

When a dataset is not included, users must obtain it independently from its official distribution channel and comply with all applicable license terms, access requirements, and usage restrictions.

Users are responsible for ensuring that their downloading, use, modification, and redistribution of any dataset complies with applicable laws and third-party intellectual property rights.


## Preparing a Calibration Dataset

You may prepare a quantization calibration dataset using either:

* A supported public dataset obtained from its official source.
* Your own representative images, which is recommended for deployment-specific calibration.

Approximately **20 representative images** are often sufficient for basic quantization testing and demonstration. However, the appropriate number may vary depending on the model, input diversity, deployment environment, and required accuracy.

For better quantization accuracy, calibration images should closely resemble the data expected during actual deployment.

For example, consider the expected:

* Lighting conditions
* Camera viewpoint
* Image resolution
* Object sizes
* Backgrounds
* Scene complexity


## Preparing the Images

Example directory structure:

```text
resources/
├── detection_dataset/
│   ├── image001.jpg
│   ├── image002.jpg
│   ├── image003.jpg
│   └── ...
└── detection_dataset.txt
```

If you are using a public dataset such as COCO 2017, select a representative subset after obtaining the dataset from its official source.

The images do not need to include annotations when they are used only for quantization calibration, unless required by the specific conversion tool.


## Configuring the Dataset List

Each dataset configuration file must contain **one image path per line**.

Example `detection_dataset.txt`:

```text
./detection_dataset/image001.jpg
./detection_dataset/image002.jpg
./detection_dataset/image003.jpg
```

Each line should point to one calibration image.

When using your own dataset, replace the example entries with paths to your images.

Relative paths are recommended because they make the configuration more portable across different systems and repository locations.

## Model and Dataset Configuration

The following table maps each dataset type to its corresponding dataset-list file.

| Target dataset      | Dataset-list file            | Included in this repository |
| :------------------ | :--------------------------- | :-------------------------  |
| COCO 2017           | `detection_dataset.txt`      |              No             |
| COCO 2017 Keypoints | `pose_dataset.txt`           |              No             |
| ImageNet 2012       | `classification_dataset.txt` |              No             |
| Total-Text          | `sign_dataset.txt`           |             Yes             |
| Custom Font Dataset | `font_dataset.txt`           |             Yes             |

Only datasets marked as included are redistributed by this repository. Configuration files for datasets that are not included demonstrate the expected path format but do not grant access to or redistribution rights for the corresponding dataset.

## Included Datasets

This repository may distribute the following calibration datasets:

### Total-Text Dataset (sign_dataset)

Selected Total-Text data is redistributed under the original **BSD 3-Clause License**.

```text
Total-Text Dataset
Copyright (c) 2018, Chee Seng Chan
All rights reserved.
SPDX-License-Identifier: BSD-3-Clause
```

The Total-Text dataset is not covered by this repository's Apache License 2.0. It remains subject to its original BSD 3-Clause License.

The complete upstream copyright notice, license conditions, and disclaimer must be included with redistributed copies.


> Copyright (c) 2018, Chee Seng Chan
> All rights reserved.
>
> Redistribution and use in source and binary forms, with or without
> modification, are permitted provided that the following conditions are met:
>
> * Redistributions of source code must retain the above copyright notice, this
>   list of conditions and the following disclaimer.
>
> * Redistributions in binary form must reproduce the above copyright notice,
>   this list of conditions and the following disclaimer in the documentation
>   and/or other materials provided with the distribution.
>
> * Neither the name of Total-Text nor the names of its
>   contributors may be used to endorse or promote products derived from
>   this software without specific prior written permission.
>
> THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS"
> AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
> IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE
> DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS BE LIABLE
> FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL
> DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR
> SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER
> CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY,
> OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE
> OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.

The upstream Total-Text project also requests that users contact the dataset author regarding commercial use. Consult the official Total-Text repository for the latest licensing and usage information.

### Custom Font Dataset (font_dataset)

The custom font calibration dataset was generated specifically for this project and is distributed as project content under the repository's Apache License 2.0, unless otherwise stated.

## Notes

* Use calibration images that represent the expected deployment inputs.
* Approximately 20 images may be sufficient for initial testing, but additional diversity can improve calibration quality.
* Ensure that every path in the dataset-list file exists and is readable.
* Use one image path per line.
* Prefer relative paths for portability.
* Do not assume that the repository's Apache License 2.0 applies to third-party datasets.
* Retain all required third-party copyright, attribution, and license notices.
* Check the official dataset source for updated license terms before downloading or redistributing a dataset.

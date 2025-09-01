# UPSID
## This is an implementation of Uncertainty-driven Progressive Single Image De-raining.
## Abstract
Over the past years, progressive methods have demonstrated promising performance in single image de-raining
task. Nonetheless, current methods still struggle to precisely remove rain and preserve more image details
during the progressive de-raining process, resulting in undesirable local artifacts or image detail loss. To tackle
these limitations, a novel progressive approach, called Uncertainty-driven Progressive Single Image De-raining
(UPSID), is proposed. Firstly, a powerful internal-and-external dense sub-network is devised, which effectively
integrates three proper and flexible components, including dense connection, long short-term memory, and
channel attention. Subsequently, the sub-network is further unfolded into multiple recurrent stages to form a
progressive de-raining network. Finally, the overall progressive de-raining network is trained with an adaptive
weighted loss to focus more on challenging pixels that characterize rain or texture/edge regions. Extensive
quantitative and qualitative experiments confirm that UPSID outperforms various state-of-the-art algorithms,
including single-stage, progressive, and uncertainty-driven single image de-raining methods. Additionally,
UPSID also demonstrates the superiority for other similar image restoration tasks such as single image
de-snowing. The code will be publicly available at https://github.com/Lcai-QZ/UPSID.
## Requirements
Python 3.8
PyTorch 2.4.1
## Quick Start
### Train
1. Train the proposed UPSID:
   ```
   Python train_UPSID.py
   ```
### Test
1. Test the proposed UPSID:
   ```
   Python test_UPSID.py
   ```

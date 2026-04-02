# pybuz
Libarary to interact with [buzcode] format.

[buzcode]: https://github.com/buzsakilab/buzcode


## Getting started

Either download or clone the repo:

```
git clone https://github.com/yfujis/pybuz/
```

Then navigate to the downloaded folder:

```
cd /path/to/pybuz
```

Install the package and requirements:

```
pip install .
```

## Example
An example code to load spike time data and construct a Pynapple object.

```
from pathlib import Path
import pandas as pd
import pynapple as nap
from pybuz.io import read_cellmetrics_cellinfo, read_spikes_cellinfo


basepath = Path('/path/to/session')

# Load Buzsáki lab (.mat) files
spikes_ = read_spikes_cellinfo(basepath)
cell_metrics = read_cellmetrics_cellinfo(basepath)

# Construct a nap.TsGroup object and attach cell_metrics as metadata
cell_metrics_df = pd.DataFrame({
    k: v for k, v in cell_metrics.__dict__.items()
    if not k.startswith('_')
})

spikes_info_df = pd.DataFrame({
    k: v for k, v in spikes_.__dict__.items()
    if not k.startswith('_')
})

spikes = nap.TsGroup([nap.Ts(times) for times in spikes_.times])

# Add cell metrics as metadata associated with each unit
spikes.set_info(cell_metrics_df)
```

## Contact

yfujishima1001[AT]gmail.com (or open an issue here).
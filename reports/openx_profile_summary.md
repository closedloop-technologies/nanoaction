# OpenX Dataset Profile Summary

Analysed datasets: asu_table_top_converted_externally_to_rlds, austin_buds_dataset_converted_externally_to_rlds, austin_sailor_dataset_converted_externally_to_rlds

### `asu_table_top_converted_externally_to_rlds`

- downloaded files: 1
- downloaded bytes: 183.5 MB
- episodes in sampled archives: 110
- sampled episodes analysed: 3
- steps per episode: min=130, avg=276.67, max=355
- episode durations (seconds): n/a
- frequent instructions: 'place down white carton' (×58), 'place down pepsi can' (×57), 'put down blue can' (×54), 'put down milk box' (×51), 'put down milk carton' (×47)

| camera | resolution (WxH) | channels | encoding |
| --- | --- | --- | --- |
| image | 224×224 | 3 | JPEG |

| signal | type | notes |
| --- | --- | --- |
| action | list[float] | numpy.ndarray shape=(7,), dtype=float32 |
| action_delta | list[float] | numpy.ndarray shape=(7,), dtype=float32 |
| language_embedding | list[float] | numpy.ndarray shape=(512,), dtype=float32 |
| state | list[float] | numpy.ndarray shape=(7,), dtype=float32 |
| state_vel | list[float] | numpy.ndarray shape=(7,), dtype=float32 |

**Archive** `asu_table_top_converted_externally_to_rlds_00000.tar` (183.5 MB)

```
├── sample_000000000000.data.pickle
├── sample_000000000001.data.pickle
├── sample_000000000002.data.pickle
├── sample_000000000003.data.pickle
├── sample_000000000004.data.pickle
├── sample_000000000005.data.pickle
├── sample_000000000006.data.pickle
├── sample_000000000007.data.pickle
├── sample_000000000008.data.pickle
├── sample_000000000009.data.pickle
├── sample_000000000010.data.pickle
└── … 99 more
```

```python
from typing import Annotated, List
from pydantic import BaseModel, Field


class AsuTableTopConvertedExternallyToRldsObservation(BaseModel):
    image: Annotated[bytes, Field(description='JPEG image 224x224 (mode=RGB)')]
    state: Annotated[list[float], Field(description='numpy.ndarray shape=(7,), dtype=float32')]
    state_vel: Annotated[list[float], Field(description='numpy.ndarray shape=(7,), dtype=float32')]


class AsuTableTopConvertedExternallyToRldsStep(BaseModel):
    action: Annotated[list[float], Field(description='numpy.ndarray shape=(7,), dtype=float32')]
    action_delta: Annotated[list[float], Field(description='numpy.ndarray shape=(7,), dtype=float32')]
    action_inst: Annotated[str, Field(description='text sample (len=8)')]
    discount: float
    goal_object: Annotated[str, Field(description='text sample (len=5)')]
    ground_truth_states: dict
    is_first: bool
    is_last: bool
    is_terminal: bool
    language_embedding: Annotated[list[float], Field(description='numpy.ndarray shape=(512,), dtype=float32')]
    language_instruction: Annotated[str, Field(description='text sample (len=17)')]
    reward: float
    observation: AsuTableTopConvertedExternallyToRldsObservation


class AsuTableTopConvertedExternallyToRldsEpisode(BaseModel):
    episode_metadata: dict
    steps: Annotated[List[AsuTableTopConvertedExternallyToRldsStep], Field(description="sampled min=130, avg=276.67, max=355")]
    image_list: list[str]
```



### `austin_buds_dataset_converted_externally_to_rlds`

- downloaded files: 1
- downloaded bytes: 288.1 MB
- episodes in sampled archives: 50
- sampled episodes analysed: 3
- steps per episode: min=639, avg=702.00, max=827
- episode durations (seconds): n/a
- frequent instructions: 'Take the lid off the pot, put the pot on the plate, and use the tool to push to pot to the front of the table.' (×2106)

| camera | resolution (WxH) | channels | encoding |
| --- | --- | --- | --- |
| image | 128×128 | 3 | JPEG |
| wrist_image | 128×128 | 3 | JPEG |

| signal | type | notes |
| --- | --- | --- |
| action | list[float] | numpy.ndarray shape=(7,), dtype=float32 |
| language_embedding | list[float] | numpy.ndarray shape=(512,), dtype=float32 |
| state | list[float] | numpy.ndarray shape=(24,), dtype=float32 |

**Archive** `austin_buds_dataset_converted_externally_to_rlds_00000.tar` (288.1 MB)

```
├── sample_000000000000.data.pickle
├── sample_000000000001.data.pickle
├── sample_000000000002.data.pickle
├── sample_000000000003.data.pickle
├── sample_000000000004.data.pickle
├── sample_000000000005.data.pickle
├── sample_000000000006.data.pickle
├── sample_000000000007.data.pickle
├── sample_000000000008.data.pickle
├── sample_000000000009.data.pickle
├── sample_000000000010.data.pickle
└── … 39 more
```

```python
from typing import Annotated, List
from pydantic import BaseModel, Field


class AustinBudsDatasetConvertedExternallyToRldsObservation(BaseModel):
    image: Annotated[bytes, Field(description='JPEG image 128x128 (mode=RGB)')]
    state: Annotated[list[float], Field(description='numpy.ndarray shape=(24,), dtype=float32')]
    wrist_image: Annotated[bytes, Field(description='JPEG image 128x128 (mode=RGB)')]


class AustinBudsDatasetConvertedExternallyToRldsStep(BaseModel):
    action: Annotated[list[float], Field(description='numpy.ndarray shape=(7,), dtype=float32')]
    discount: float
    is_first: bool
    is_last: bool
    is_terminal: bool
    language_embedding: Annotated[list[float], Field(description='numpy.ndarray shape=(512,), dtype=float32')]
    language_instruction: Annotated[str, Field(description='text sample (len=110)')]
    reward: float
    observation: AustinBudsDatasetConvertedExternallyToRldsObservation


class AustinBudsDatasetConvertedExternallyToRldsEpisode(BaseModel):
    episode_metadata: dict
    steps: Annotated[List[AustinBudsDatasetConvertedExternallyToRldsStep], Field(description="sampled min=639, avg=702.00, max=827")]
    image_list: list[str]
```



### `austin_sailor_dataset_converted_externally_to_rlds`

- downloaded files: 2
- downloaded bytes: 1.4 GB
- episodes in sampled archives: 99
- sampled episodes analysed: 6
- steps per episode: min=484, avg=1408.33, max=2377
- episode durations (seconds): n/a
- frequent instructions: 'Interact with the objects in diverse but meaningful ways.' (×6845), 'Place the fish, sausage, and tomato into the frying pan.' (×980), 'Place the pan onto the stove and place the fish and sausage into the pan.' (×625)

| camera | resolution (WxH) | channels | encoding |
| --- | --- | --- | --- |
| image | 128×128 | 3 | JPEG |
| wrist_image | 128×128 | 3 | JPEG |

| signal | type | notes |
| --- | --- | --- |
| action | list[float] | numpy.ndarray shape=(7,), dtype=float32 |
| language_embedding | list[float] | numpy.ndarray shape=(512,), dtype=float32 |
| state | list[float] | numpy.ndarray shape=(8,), dtype=float32 |
| state_ee | list[float] | numpy.ndarray shape=(16,), dtype=float32 |
| state_gripper | list[float] | numpy.ndarray shape=(1,), dtype=float32 |
| state_joint | list[float] | numpy.ndarray shape=(7,), dtype=float32 |

**Archive** `austin_sailor_dataset_converted_externally_to_rlds_00000.tar` (729.4 MB)

```
├── sample_000000000000.data.pickle
├── sample_000000000001.data.pickle
├── sample_000000000002.data.pickle
├── sample_000000000003.data.pickle
├── sample_000000000004.data.pickle
├── sample_000000000005.data.pickle
├── sample_000000000006.data.pickle
├── sample_000000000007.data.pickle
├── sample_000000000008.data.pickle
├── sample_000000000009.data.pickle
├── sample_000000000010.data.pickle
└── … 36 more
```

**Archive** `austin_sailor_dataset_converted_externally_to_rlds_00001.tar` (723.8 MB)

```
├── sample_000000000047.data.pickle
├── sample_000000000048.data.pickle
├── sample_000000000049.data.pickle
├── sample_000000000050.data.pickle
├── sample_000000000051.data.pickle
├── sample_000000000052.data.pickle
├── sample_000000000053.data.pickle
├── sample_000000000054.data.pickle
├── sample_000000000055.data.pickle
├── sample_000000000056.data.pickle
├── sample_000000000057.data.pickle
└── … 41 more
```

```python
from typing import Annotated, List
from pydantic import BaseModel, Field


class AustinSailorDatasetConvertedExternallyToRldsObservation(BaseModel):
    image: Annotated[bytes, Field(description='JPEG image 128x128 (mode=RGB)')]
    state: Annotated[list[float], Field(description='numpy.ndarray shape=(8,), dtype=float32')]
    state_ee: Annotated[list[float], Field(description='numpy.ndarray shape=(16,), dtype=float32')]
    state_gripper: Annotated[list[float], Field(description='numpy.ndarray shape=(1,), dtype=float32')]
    state_joint: Annotated[list[float], Field(description='numpy.ndarray shape=(7,), dtype=float32')]
    wrist_image: Annotated[bytes, Field(description='JPEG image 128x128 (mode=RGB)')]


class AustinSailorDatasetConvertedExternallyToRldsStep(BaseModel):
    action: Annotated[list[float], Field(description='numpy.ndarray shape=(7,), dtype=float32')]
    discount: float
    is_first: bool
    is_last: bool
    is_terminal: bool
    language_embedding: Annotated[list[float], Field(description='numpy.ndarray shape=(512,), dtype=float32')]
    language_instruction: Annotated[str, Field(description='text sample (len=57)')]
    reward: float
    observation: AustinSailorDatasetConvertedExternallyToRldsObservation


class AustinSailorDatasetConvertedExternallyToRldsEpisode(BaseModel):
    episode_metadata: dict
    steps: Annotated[List[AustinSailorDatasetConvertedExternallyToRldsStep], Field(description="sampled min=484, avg=1408.33, max=2377")]
    image_list: list[str]
```



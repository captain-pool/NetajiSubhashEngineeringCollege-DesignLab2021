## Aggregated Sentiment Analysis of Hotel Reviews


#### Prompt
*Analyse the sentiment of hotel reviews and also give the average sentiment related to certain specific words.*

**Example**
- The location is good but the staffs are not cordial
- The location is great and the cleanliness is also quite good but the staff are not very cooperative.
- Everything is good

Now, depending on these reviews you got to analyse it’s:

* Location - `8` / Good
* Cleanliness -  `6` / Not bad
* Hospitality - `3` / Can be improved
* `<insert aspect here>` - `<score>`

### Usage Instructions
Install Miniconda (or Anaconda): https://docs.conda.io/projects/conda/en/latest/user-guide/install/
```bash
$ conda env create -f environment.yml
$ cd final_assignment # or `practical_assignment`
$ python3 query.py
```

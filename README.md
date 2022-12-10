# Biometric system based on behavioral biometrics: typing errors in English. 

## General info
Project includes biometric system based on typing errors in English created in Python that allows to gather information from users in real time and from text file. 
To ensure clear separation of tasks, the biometric tool relies on **generic biometric system model**. 

### Technologies


## Concept
The concept of system assume the following phases:
1. **Data acquisition**
2. **Signal processing**
    - Pre-processing
    - Feature extraction
    - Quality control
3. **Matching**
4. **Decision**

Each of phase that is contained in concept of biometric system based on generic model is implemented as at least one separated module.


### Data acquisition
For now the program joins features of keylogger and spellchecker gathering informations to the .json file. 
System bases on **contexts** which are similar to sentences - in Online mode context is finished whenever user 
### 

Spellchecking bases on [language-tool-python](https://pypi.org/project/language-tool-python/). 


### Signal processing

#### Pre-processing

#### Feature extraction

#### Quality control

### Matching

### Decision making


## How to get started

Due to the willness to handle each case, the [pynput](https://pypi.org/project/pynput/) library was extended to support the numeric keypad. **This tool will not work properly until the _pynput.keyboard._win32_ would not be updated with numeric keys.**

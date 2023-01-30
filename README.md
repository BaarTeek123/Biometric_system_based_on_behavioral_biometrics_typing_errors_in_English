# Biometric system based on behavioral biometrics: typing errors in English. 

## General info

Project includes biometric system based on typing errors in English created in Python that allows to gather information from users in real time and from text file. The biometric tool was written in the Python 3.9 using object-oriented design and available libraries, packages and modules, e.g. [Natural Language Toolkit (NLTK)](https://www.nltk.org/) - the set of libraries and models for natural language processing or [pynput](https://pypi.org/project/pynput/). 




<!-- This system joins features of spellechecker and keylogger (in _Online_ mode) to extract string metrics and error types that can be use as a biometric features.  -->

### Requirements
Most of required libraries are included in [requirements.txt](https://github.com/BaarTeek123/Biometric-system-based-on-behavioral-biometrics-typing-errors-in-English.-/blob/master/requirements.txt) file. 

Due to the willness to handle each case, the [pynput](https://pypi.org/project/pynput/) library was extended to support the numeric keypad. 
**This tool will not work properly until the _pynput.keyboard.\_win32_ would not be updated with numeric keys.** 
The updated _Key class_ is saved in the [text file](https://github.com/BaarTeek123/Biometric-system-based-on-behavioral-biometrics-typing-errors-in-English.-/blob/master/edited_key_class_pynput.keyboard_win32) - there is need to replace the original _Key_ class from  _pynput.keyboard.\_win32_ by the extended class. 


## Concept
<!-- [concept](https://user-images.githubusercontent.com/59124934/206584979-852fb33d-36d7-4841-aebe-e1029af5e449.png) -->

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

System bases on **_contexts_** which are similar to sentences, but data collection varies depending on the selected system mode. The output of this phase is _context_. 





#### _Online_ mode



In _Online_ mode a context is finished whenever user click a key that can assume a finish of a piece of text or an idea, e.g.  _._, _?_, _\n_ or _!_, but also a left mouse key or types the _up_ or _down_ _arrow key_. The last three end the context, because a user may , e.g. switch line using _up_ or _down_ _arrow_  or switch the document using a left key mouse. 

The system also defined variable _cursor_ representing current position of the cursor which can be changed using _left_ or _right array_ key. The maximum value of this variable is 0 and it means that each typed character would be append at the end of context. However if the value is -2, the typed character will be added in second to the last. 

For example, the following sentence is gathered: _I love co_ and the user typed _d_: 
- if the _cursor_ = 0, the sentence will be like  _I love cod_
- if the _cursor_ = -3, the sentence will be like  _I loved co_

_Cursor variable_ is used also whenever user clicks _Delete_ or _Backspace_. In case clicking _Backspace_ the char before the _cursor_ variable is deleted, in case _Delete_ the character behind the _coursor_ (but the value **must be < 0**). For example the sentence is as follows:  _I love codding in_ and: 
- if the _cursor_ = 0 and user clicks:
    - _Delete_ - nothing will happen.
    - _Backspace_ - the last character will be deleted - the senstence will be _I love codding i_
- if the _cursor_ = -4 and user clicks:
    - _Delete_ - the character behind the _coursor_ will be deleted - the senstence will be _I love **coddig** in_. 
    - _Backspace_ - the character preceding the _coursor_ will be deleted - the senstence will be _I love **coddng** in_.

In case clicking any of keys that may be assumed as changing the line or the page (like left mouse key), _cursor_ value is zeroed. 




#### Offline mode


Offline mode bases on uploaded text file that is split into contexts basing on puntuation, like _._, _?_ or _\n_. 




### Signal processing


The input of this phase is _context_ (in _Offline_ mode it is being iterated through each of them). The output is a user model based on _n-grams_. The _n-grams_ bases on misspelled word representation using _k_ features. 


#### Pre-processing


This phase starts with punctuation clearing the original _context_. The cleaned one is provided to a spellchecker - [language-tool-python](https://pypi.org/project/language-tool-python/). The output of spellchecker and the original _context_ is split into words and Part-of-Speech tags (POS-tags) are assigned to each of them. The output of this phase are two lists:
1. List of tokens (words) from the original _context_ with assigned POS-tags;
2. List of tokens (words) from the corrected _context_ (result of spellchecker) with assigned POS-tags;


#### Feature extraction #1


For each of original token occurs comparison with those from corrected sentence. In the case of discrepancies between them, the original and corrected token is provided to the _Word_ class that represents misspelled words. Then follows a calculation of choosen string metrics from _Distances_ class. The string metrics can be grouped into following categories:
- edit distance;
- token based; 
- sequence based;
- phonetic based;

Moreover, it was created a superclass _Edit Operation_ represents a base for each subclass representing operation type basing on operation types defined in [Damerau-Levensthein](https://bmcbioinformatics.biomedcentral.com/articles/10.1186/s12859-019-2819-0)
- _Transpose_;
- _Delete_; 
- _Replace_;
- _Insert_. 
These classes provide more context about typing errors, like wrong keys, a characater preceding or following the error. 

#### Quality control
Quality control bases on [Damerau-Levensthein](https://bmcbioinformatics.biomedcentral.com/articles/10.1186/s12859-019-2819-0) distance similarity: 
- for strings that length < 3:
- for for strings that length >= 3:
The threshold is set due to imperfection of spelling correction. If the distance is too high, the corrected word will be considered as incorrect due to the large differences between the orignal and corrected. 


#### Feature extraction #2
**Before creating _n-grams_ the set of misspelled word representations has to be splited into 2 separated - for training and testing processes.**
For each of those sets of _k_-feature misspelled word representations is created a set _n-grams_ of _k_-feature word representation.
<p align="center">
    <img src="https://user-images.githubusercontent.com/59124934/207051467-bd0af2ff-426b-4f4f-ba9c-e3a3dd1cda3a.png">
</p>
Those sets are labeled and assigned for each class (user) and will be used as an input for the next phase. 



<!-- ### Matching -->



<!-- ### Decision -->



<!-- ## Implementation
### Data Gathering
#### _Online_ mode 
<p align="center">
 <img src="https://user-images.githubusercontent.com/59124934/206594360-6bbfaa04-7056-4096-91b2-d491077d4aa4.jpg">
</p>

#### _Offline_ mode
<p align="center">
 <img src="https://user-images.githubusercontent.com/59124934/206594336-2331fdf2-873d-4dcf-a4a1-d3b2205f71bd.jpg">
</p>

### Signal processing

#### Pre-processing


<p align="center">
 <img src="https://user-images.githubusercontent.com/59124934/206594269-20a2da08-3433-4115-802e-a3ee4865898f.jpg">
</p>

#### Feature extraction
<p align="center">
 <img src="https://user-images.githubusercontent.com/59124934/206594278-1e64ca8a-2bf2-448d-b2cf-368a0ca50a79.jpg">
</p>
#### Quality control

### Matching

### Decision making -->






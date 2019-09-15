## Satterthwaite Data README

Demographic and clinical survey/test data, $n = 7151$ but precursor to imaging studies so not all were imaged. Functional and structural networks. Functional networks include  Structural networks include deterministic networks (n=1100) (FA, ADC, streamline length, streamline count) and probabilistic networks (n=1094, 6 not imaged).

See listed papers at the end for further details.

The folder structure of the available data is as follows. A single file name with a pair of "{}" denotes a list of files with the brackets encasing the variable portion of the name.

```
/Collab/
└── PNC_Data_9498/
  └── Cognitive_MedicalFilter.csv
  └── Demographics_MedicalFilter.csv
  └── GOA_MedicalFilter.csv
  └── GOA_imputed_MedicalFilter.csv
└── Imaging/
  └── FC/
    └── REST_Alone/
      └── SubjectsIDs_Schaefer_rest_Alone.csv
      	// scanid <-> bblid, scan metadata
      └── REST_Data/
        └── {scanid}_Schaefer400_network.txt
        	//1015 subjects, size=(400,400)
    └── REST_Task/
      └── SubjectsIDs_Schaefer_rest_task.csv
      	// scanid <-> bblid, scan metadata
      └── REST_Data/
        └── {scanid}_Schaefer400_network.txt
        	//842 subjects size=(400,400)
      └── Idemo_Data
        └── {x}_{y}x{z}_SchaeferPNC_network.txt
        	//844 subjects size=(400,400)
      └── NBack_Data
        └── {x}_{y}x{z}_schaefer400_network.txt
        	//845 subjects size=(79800,)
    └── SC/
      └── SubjectsIDs_Schaefer_Diffusion.csv
      └── Deterministics_ADC/
        └── {x}_{y}x{z}_SchaeferPNC_400_dti_adc_connectivity.mat
        	\\1105 subjects: sparse (400,400) connectivities, names
      └── Deterministic_FA/
        └── {x}_{y}x{z}_SchaeferPNC_400_dti_fa_connectivity.mat
        	\\1101 subjects: sparse (400,400) connectivities, names
      └── Probabilistic
        └── {x}_{y}x{z}_ptx_p1000_wmEdge_Schaefer400_17net.mat
        	\\1095 subjects: volumes, streamlines, connections
      └── Deterministic_SC
        └── {x}_{y}x{z}_SchaeferPNC_400_dti_streamlineCounts_connectivity.mat
        	\\1103 subjects: sparse (400,400) connectivities, names
      └── Deterministic_Length
        └── {x}_{y}x{z}_SchaeferPNC_400_dti_mean_streamlineLength_connectivity.mat
        	\\1102 subjects: sparse (400,400) connectivities, names
```

* **Cognitive_MedicalFilter.csv**
  
  14 tests assessing 5 neurobehavioral functions (executive control, episodic memory, complex cognition, social cognition, sensorimotor speed)
  
  * Categorical, Continuous
  * Missing data
  * zscores (larger value always denote better performance)
  * Sensitive to age and sex
  
* **Demographics_MedicalFilter.csv**
  
  * Demographic info
  * Categorical, Continuous
  
* **GOA_MedicalFilter.csv**
  
  * Categorical
  * Missing Data
  * GOASSES scores assessing of mood, anxiety, behavioral, eating disorders and psychosis spectrum symptoms and substance use history
  
* **GOA_imputed_MedicalFilter.csv**
  
  * Same as GOA_MedicalFilter.csv but with imputed scores (unknown mechanism)

#### FC/REST_Alone/

1015 resting-state functional connectivities

#### FC/REST_Task/

842 subjects with three functional data (resting state, Nback fMRI, emotion identification fMRI)

Each matrix file named by the subjects' BBLID and SCANID (found in excels)

## Relevant Papers/Links

* Gur RC, Richard J, Calkins ME, Chiavacci R, Hansen JA, Bilker WB, Loughead J, Connolly JJ, Qiu H, Mentch FD, Abou-Sleiman PM, Hakonarson H, Gur RE. **Age group and sex differences in performance on a computerized neurocognitive battery in children age 8–21**. Neuropsychology. 2012;26:251–265.
* Calkins ME, Merikangas KR, Moore TM, et al. **The Philadelphia Neurodevelopmental Cohort: constructing a deep phenotyping collaborative**. *J Child Psychol Psychiatry*. 2015;56(12):1356–1369. doi:10.1111/jcpp.12416
* Satterthwaite TD, Elliott MA, Ruparel K, et al. **Neuroimaging of the Philadelphia neurodevelopmental cohort**. *Neuroimage*. 2014;86:544–553. doi:10.1016/j.neuroimage.2013.07.064
* Xia CH, Ma Z, Ciric R, et al. **Linked dimensions of psychopathology and connectivity in functional brain networks.** *Nat Commun*. 2018;9(1):3003. Published 2018 Aug 1. doi:10.1038/s41467-018-05317-y
* Data Website: <https://www.med.upenn.edu/bbl/philadelphianeurodevelopmentalcohort.html>




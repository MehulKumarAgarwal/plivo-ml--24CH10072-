1190 ms at 5.0% cutoffs, AUC 0.599
I replaced the starter functions with pitch slope, trailing energy and voicing fraction to better capture the intonation of conversational "holds".



Score: 508 ms @ 5.0% cutoffs (AUC: 0.967)
Replaced Logistic Regression with a StandardScaler + RandomForestClassifier pipeline. The non-linear model drastically improved the separation of classes based on pitch and energy dynamics.

Score: 850 ms @ 5.0% cutoffs on Hindi (AUC: 0.551).
Note: Tested the saved Random Forest model on the unseen Hindi dataset. The model successfully beats the 1600 ms baseline, but the drop in AUC highlights the challenge of cross-lingual generalization when training solely on English prosody.


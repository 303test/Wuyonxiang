
%load('C:\Users\ljm\Desktop\BDCA\BDCA²âÊÔ½á¹û\BDCA_runover.mat');
lab=EEG.bdca.labels(:);
pot=EEG.bdca.pot(:);
AZ=0;
for epo = 1:EEG.trials
  if pot(epo)<0
      pr_lab(epo)=0;
  else
      pr_lab(epo)=1;
  end
  
  A=lab(epo)-pr_lab(epo);
  if A==0
      AZ=AZ+1;
  end
  
end
AZ
Az=AZ/EEG.trials




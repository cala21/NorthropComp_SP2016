#Models/varients to run

##Models
- get_unet
- get_seq
- get_gnet
##Configuration
These are found in configuration.txt
- N_subimgs (needs to be a multiple of 77) have it larger than 100,000 - probably like 308,000
- N_epochs Set this to something between 500 and 2000
- patch_{height|width} I have no idea what we should adjust this to, if anything
- N_classes see what happens when we run this with the full number of classes (6 total classes)

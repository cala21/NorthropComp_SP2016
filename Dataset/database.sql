CREATE DATABASE IF NOT EXISTS `goes`;
use `goes`;
CREATE TABLE IF NOT EXISTS `goes_data` (Id Integer, PixelData MEDIUMBLOB , date datetime, PixelLabels MEDIUMBLOB, NextFrame Integer, PrevFrame Integer);

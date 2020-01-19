library(tidyverse)
library(survey)

rm(list = ls())

yrbs <- read_csv("./Dev/survey-methods/survmeth/tests/estimation/yrbs.csv")
yrbs <- yrbs %>%
  select(-X1)

head(yrbs, 30)


yrbs_design <- svydesign(id = ~psu , data = yrbs , weight = ~weight , strata = ~stratum)

yrbs$qn8[yrbs$qn8 == 2] <- 0 
#yrbs$qn8[is.na(yrbs$qn8)] <- 0 
svymean(qn8,yrbs_design, na.rm = TRUE, deff = TRUE)
head(yrbs, 100)

yrbs_design <- svydesign(id = ~psu , data = yrbs , weight = ~weight)

yrbs$qn8[yrbs$qn8 == 2] <- 0 
svymean(qn8,yrbs_design, na.rm = TRUE, deff = TRUE)

svyciprop(~qn8, yrbs_design, method = "lo")
svyciprop(~qn8, yrbs_design, method = "li")

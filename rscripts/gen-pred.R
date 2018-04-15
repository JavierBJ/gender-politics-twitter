library(gender)
politicians <- c('dumpusers01t.csv', 'dumpusers02t.csv', 'dumpusers03t.csv', 
                 'dumpusers04t.csv', 'dumpusers05t.csv', 'dumpusers06t.csv', 
                 'dumpusers07t.csv', 'dumpusers08t.csv', 'dumpusers09t.csv', 
                 'dumpusers10t.csv', 'dumpusers11t.csv', 'dumpusers12t.csv', 
                 'dumpusers13t.csv')
people <- c('dumpusers01mr.csv', 'dumpusers02mr.csv', 'dumpusers03mr.csv', 
                 'dumpusers04mr.csv', 'dumpusers05mr.csv', 'dumpusers06mr.csv', 
            'dumpusers07mr.csv', 'dumpusers08mr.csv', 'dumpusers09mr.csv', 
            'dumpusers10mr.csv', 'dumpusers11mr.csv', 'dumpusers12mr.csv', 
            'dumpusers13mr.csv')
names <- c('aux')
for (f in people) {
  data <- read.csv(f, sep=';')
  these.names <- as.character(data$name)
  these.names <- gsub(" .*$", "", these.names)
  names <- c(names, these.names)
}
names <- unique(names)
res <- gender(names, years=c(2000,2012))
print(length(names))
print(nrow(res))
#View(res)
res <- res[,c('name', 'proportion_male')]
write.csv(res, file='r-genders.csv')
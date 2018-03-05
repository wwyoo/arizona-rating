#Author: William Weimin Yoo
#Preprocessing of raw user ratings data
#Processed data will be imported to MATLAB
#Aim is to transform raw data entries to a matrix of row (user) and
#column (business) indices with associated stars (ratings 1 to 5).
#This matrix will then be converted to sparse form in MATLAB
#for more efficient computations

#load package to do json data transfer
library(RJSONIO)

#function to convert json to R data frame
process <- function(filename){
  data=file(filename,"r")  #make connection (read only)
  djson=readLines(data)  #read all json data
  m=length(djson)
  dlist=vector("list",m)  #save as list at intermediate step

  for(i in 1L:m){
    dlist[[i]]=fromJSON(djson[i])  #json -> R (list) for each row
  }

  output=as.data.frame(t(sapply(dlist,rbind))) #convert to R data frame
  names(output)=names(dlist[[1L]])
  close(data)  #close connection
  return(output)
}

#yelp data conversion
business=process("yelp_academic_dataset_business.json")
save(business,file="business.RData")
#11537 businesses total

checkin=process("yelp_academic_dataset_checkin.json")
save(checkin,file="checkin.RData")
#only 8282 businesses have checkins

#our aim is review prediction and we will mainly
#use this data set
review=process("yelp_academic_dataset_review.json")
save(review,file="review.RData")
#229907 reviews

user=process("yelp_academic_dataset_user.json")
save(user,file="user.RData")
#43873 users

#to buid user-business rating matrix,
#we just need user id, business id and ratings
reviework=review[,c("user_id","stars","business_id")]

#Remove businesses with less than 10 reviews
bid=reviework[,"business_id"]
#each column in data frame is a list, unlist for easier processing
bid=unlist(bid)
count.bus=table(bid)
bidx=which(count.bus < 10L)
cbus=which(bid%in%names(bidx))
reviework2=reviework[-cbus,]

#Remove users with less than 20 reviews
uid=reviework2[,"user_id"]
uid=unlist(uid)
count.user=table(uid)
uidx=which(count.user < 20L)
cuser=which(uid%in%names(uidx))
reviework3=reviework2[-cuser,]

#data analysis will be done in MATLAB, and the review data
#will be stored as sparse matrix, hence we need to sort the
#data by business then user and convert their id's to integers
#representing rows and columns
bids=reviework3[,"business_id"]
bids=unlist(bids)
review.sort.bus=reviework3[order(bids),]
bid.sort=unlist(review.sort.bus[,"business_id"])
colind=cumsum(1-duplicated(bid.sort))
review.sort.bus$col=colind

uids=review.sort.bus[,"user_id"]
uids=unlist(uids)
review.sort.user=review.sort.bus[order(uids),]
uid.sort=unlist(review.sort.user[,"user_id"])
rowind=cumsum(1-duplicated(uid.sort))
review.sort.user$row=rowind

#row and column numbers for sparse matrix in MATLAB
#with associated ratings
user.row=review.sort.user[,"row"]
bus.col=review.sort.user[,"col"]
stars=review.sort.user[,"stars"]
stars=unlist(stars)

#store as matrix
reviewdat=cbind(user=user.row,business=bus.col,stars=stars)

#save in ASCII txt to be read using MATLAB
write(t(reviewdat),file="reviewdat.txt",ncolumns=3L)

#on to MATLAB -> see MATLAB file matrixcomplete.m

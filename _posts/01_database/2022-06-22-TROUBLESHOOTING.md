---
layout: single
title:  DATABASE> Troubleshooting (Delete error), (DELETE, TRUNCATE, DROP의 차이)
categories: 01_database
tag: [database, TRANSACTION, LOG, DROP, DELETE, TRUNCATE]
toc: true
author_profile: false
sidebar:
    nav: "docs"
---



server를 손대는건 참 무섭다. 내 data도 아니고 회사 data인데, 그리고 선무당이 사람 잡는다고 나처럼 주먹구구식으로 computer programming을 배운 놈이 만지는건 내가 생각해도 참 무섭다. 하지만 어쩌겠는가 일은 해야하는 법. 오늘도 난 그냥 해본다. 

original server(O.S)와 backup server(B.S)가 있다. 

# 1. 문제가 뭐냐

문제: O.S에서 B.S로 하루에 한 번 data를 export한다. 하루에 2Mrows(200만줄)을 export하는데 한달이면 15G정도 차지함. 기존 서버 용량이 220G정도 되어 1.5년 정도 되니 꽉 차서 query도 잘 안보내짐. 그래서 주기적으로 삭제를 해줘야 했음.

참고로 아래에서는 O.S의 DB가 SeojunAndon이고, B.S의 DB가 AKANDON이다. 

```sql
#####1.이걸로 database지우고####################
DELETE 
FROM PickupDetail 
where dtCreated < '2021-02-06 03:12:18.443'
###############################################

#####2.이걸로 transactionlog지우고###############
USE [AKANDON];
GO
 
ALTER DATABASE [AKANDON]
SET RECOVERY SIMPLE;
GO
 
DBCC SHRINKFILE ( [SeojunAndon_log] , 1);
GO
 
ALTER DATABASE [AKANDON]
SET RECOVERY FULL;
GO
###################################################

########3.이걸로 mdb file 지우고#########################
dbcc shrinkfile(SeojunAndon)
####################################################

#########4.앞에서 너무 많은 양을 지우다가 recovery pending 에러 나서 아래로 recovery 했는데 정석적인 방법은 아니란다#####

ALTER DATABASE [AKANDON]  SET EMERGENCY;

GO

ALTER DATABASE [AKANDON]  SET SINGLE_USER;

GO

DBCC CHECKDB (AKANDON, REPAIR_ALLOW_DATA_LOSS) WITH ALL_ERRORMSGS;

GO

ALTER DATABASE [AKANDON] SET MULTI_USER;

GO
########################################################################################################
```

code들에 대한 설명은 아래와 같다. 

1. 2~3일치를 삭제: 이걸 하고 나면 TR log가 증가한다.
2. Transaction log 삭제
3. mdf file 삭제

어느 날은 보통 2~3일치를 줄이는걸 용량이 많이 확보되어 10일치를 지웠더니 일정 시간 후 management studio가 멈췄다. 강제로 종료할 때 task manager로 삭제했는데 그 다음부터 index가 꼬이는 error가 발생해서 delete가 안먹혔다.

# 2. 어떻게 해결했나

업체에서는 drop table을 해야 한다고 했는데 찾아보니 truncate이 있어서 이 방법으로 해보기로 함. 

최근 3주 data는 index가 안꼬였을거라 그냥 믿어보고 그걸 다른 서버(O.S)로 export 하고, truncate table을 하였다. 그러니 용량이 150G 이상 확보되었다.
앞서 O.S로 옮긴 data를 다시 B.S로 옮겼다. 이상없이 끝났다.

# 3. Table을 지우는데는 세가지 방법(delete / truncate / drop)이 있다.
<img src = "/assets/img/bongs/trouble.PNG">
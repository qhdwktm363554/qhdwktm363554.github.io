---
layout: single
title:  MYSQL> Hackerrank 풀이 - Challenges (medium level) (complicated join concept)
categories: 01_database
tag: [database, QUERY, JOIN, INNER, OUTER, FULL OUTER]
toc: true
author_profile: false
sidebar:
    nav: "docs"
---

굳이 이 문제에 대한 풀이를 남겨놓는 이유는 이 문제가 지금까지 hackerrank coding test 하면서 가장 오래 걸렸기도 하고, 개념적으로 중요한 부분이라고 생각해서 남긴다. 시간이 날 때 다시 봐보자. 정말 눈알이 돌아가게 복잡하다.  basic join이라고 해서 사실 쉽게 생각했는데 절대 basic join이 아니였다. 조건절에 조건을 주기 위해 값 하나를 얻어야 했고, 이를 위해 이중쿼리를 작성해야 했다. 처음에는 완전히 이해 이해할 수 없을 개념이라고 생각했지만 보다보니 이해가 가고 내가 직접 코드를 작성할 수 있었다. 

문제> 

<img src = "/assets/img/bongs/220621/pic1.png">

<img src = "/assets/img/bongs/220621/pic2.png">

우선 아래와 같은 code를 작성하긴 했는데 그 다음에 어떻게 having clause에 조건을 줘야 할 지 몰랐다. 

```sql
select H.hacker_id, H.name, count(C.challenge_id) as C_created
from Hackers H 
     INNER JOIN Challenges C on H.hacker_id = C.hacker_id
group by   H.hacker_id, H.name
having 
    C_created = 
   OR 
   C_created IN 
order by C_created desc, H.name;
select Ha.hacker_id, Ha.name, count(Ch.challenge_id) as Challenge_created
from Hackers Ha INNER JOIN Challenges Ch on Ha.hacker_id = Ch.hacker_id group by Ha.hacker_id, Ha.name
```

그래서 주석을 달아보았다. 

```sql
select H.hacker_id, H.name, count(C.challenge_id) as C_created
from Hackers H 
     INNER JOIN Challenges C on H.hacker_id = C.hacker_id
group by   H.hacker_id, H.name
having 
   -- 조건이 들어가야 하는데 C_created가 최대 count(C_created)와 같을 때
   -- 하나의 값만 return 하는 값이 들어가야 한다. 그러려면 이중쿼리를 써야 하겠다.
    C_created = 조건1
   OR 
   -- 조건이 들어가야 하는데 C_created가 count(C_created)가 1보다 큰 것
   C_created IN (조건2)
order by C_created desc, H.name;
select Ha.hacker_id, Ha.name, count(Ch.challenge_id) as Challenge_created
from Hackers Ha INNER JOIN Challenges Ch on Ha.hacker_id = Ch.hacker_id group by Ha.hacker_id, Ha.name
```

하지만 모르겠다.  아래 1번과 2번 query를 모두 위의 조건에 각각 넣었다. 

1. max값으로 들어갈 단일 record를 return하는 query를 만들어보았다. 

```sql
select max(NEW.Challenge_created)
                    from (
                            select Ha.hacker_id, Ha.name, count(Ch.challenge_id) as Challenge_created
                            from Hackers Ha INNER JOIN Challenges Ch on Ha.hacker_id = Ch.hacker_id 
                            group by Ha.hacker_id, Ha.name
) NEW; 
```

1. IN 함수 안에 들어갈 여러 record를 return하는 query를 만들어보았다. 

```sql
select NEW_2.Challenge_created_2
                    from (
                            select Ha.hacker_id, Ha.name, count(Ch.challenge_id) as Challenge_created_2
                            from Hackers Ha INNER JOIN Challenges Ch on Ha.hacker_id = Ch.hacker_id 
                            group by Ha.hacker_id, Ha.name
                        ) NEW_2 group by NEW_2.Challenge_created_2 having count(*) = 1; 
```

그리고 최종 완성된 query 

```sql
select H.hacker_id, H.name, count(C.challenge_id) as C_created
from Hackers H 
    INNER JOIN Challenges C on H.hacker_id = C.hacker_id
group by   H.hacker_id, H.name
having 
    C_created = (select max(NEW.Challenge_created)
                    from (
                            select Ha.hacker_id, Ha.name, count(Ch.challenge_id) as Challenge_created
                            from Hackers Ha INNER JOIN Challenges Ch on Ha.hacker_id = Ch.hacker_id 
                            group by Ha.hacker_id, Ha.name
) NEW )
    OR 
    C_created IN (select NEW_2.Challenge_created_2
                    from (
                            select Ha.hacker_id, Ha.name, count(Ch.challenge_id) as Challenge_created_2
                            from Hackers Ha INNER JOIN Challenges Ch on Ha.hacker_id = Ch.hacker_id 
                            group by Ha.hacker_id, Ha.name
                        ) NEW_2 group by NEW_2.Challenge_created_2 having count(*) = 1
                         )
order by C_created desc, H.hacker_id;
```

드디어 감격의 Congratulations!를 받았다. 그동안 정말 힘들었다 ㅋㅋ 아오 어떻게 query를 짜야 하는지 알았어도 긴 query를 직접 작성하니 syntax error 도 많이나고 생각보다 너무 빡셌다. 이틀동안 이것만 보고 있었던건 아니지만 어쨌건 이틀이나 소요되었다. 

<img src = "/assets/img/bongs/220621/pic3.png">
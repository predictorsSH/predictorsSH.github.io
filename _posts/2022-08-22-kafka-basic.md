---
layout: post
title: kafka 아키텍처 설명
subtitle: Grad-CAM Paper
author: san9hyun
categories: kafka 
banner : /assets/images/banners/horses.jpg
tags: kafka study
---

데브원영님의 카프카 강의를 듣고 정리함 <br>
강의는 인프런에서 무료로 제공하고 있고, 데브원영님의 유튜브 채널에서도 볼 수 있다!

혼자 공부한 것을 기록 한 것이라, 잘못된 내용이 있을 수도 있음.

## 카프카란?
"Apache Kafka is an open-source distributed event streaming platform used by thousands of companies for high-performance data pipelines, streaming analytics, data integration, and mission-critical applications."<br>
카프카 공식 웹 사이트에서 만나 볼 수 있는, 아파치 카프카에 대한 설명이다.<br>

정리해보면,<br>
아파치 카프카란 데이터 파이프라인 구축, 스트리밍 분석, 데이터 통합을 수행할때 사용할 수 있는 분산 메시지 스트리밍 플랫폼이다. 

카프카의 장점이자 특징은 아래와 같다.<br>
- 고가용성
- 확장성
- 분산처리


## 카프카 아키텍처


![kafka-architecture](/assets/images/contents/kafka/kafka_architecture.PNG)



## 구성요소별 설명

- Broker : kafka를 구성하는 각 서버, 서버 1대당 Broker 1개라고 보면된다.
- Topic : 특정 데이터를 저장(관리)하는 하나의 그룹, 하나의 토픽에 여러개의 파티션이 존재하여, 파티션 수 만큼 분산처리
- Partition : 각 토픽당 데이터를 분산처리하는 단위, 하나의 큐라고 볼 수 있음. Producer는 데이터를 각 partition 큐에 적재한다. partition 개수를 카프카 운용도중 늘릴 수는 있지만, 줄일 수는 없다고 한다! 줄일려면 토픽을 제거하는 수 밖에 없다. 그렇게 되면 데이터 유실이 발생할 것이다.
- replication : 데이터 유실 방지를 위한 복제. **Broker 수만큼 복제할 수 있다.** 특정 Broker에 장애가 발생하더라도 서비스가 중단 되지 않는다. 
- Producer(publisher) : 데이터를 발생시키고 카프카 클러스터에 적재
- Consumer(subscriber) : 컨슈머가 partition에 적재되어있는 데이터를 읽고 처리한다. 컨슈머 그룹안의 컨슈머 수 만큼 partition의 데이터를 분산처리 할 수 있다. 또한 하나의 컨슈머가 문제가 생기더라도, 그룹내 다른 컨슈머에 의해 계속해서 데이터를 처리할 수 있다.
- Zookeeper : kafka 운용을 위한 coordination service, 카프카 브로커들을 하나의 클러스터로 관리하기 위해서 필요. 클러스터에 속한 서버들 끼리 정보 공유할 수 있도록 해줌.

Zookeeper는 아파치 카프카에서 제거될 것이라고 한다. <br>

## offset

데이터 유실방지를 위해 kafka에는 offset을 따로 저장한다.<br>
partition 큐에 들어있는 데이터 중 몇번까지 읽었는지를 offset이란 이름으로 저장하는 것이다.<br>

offset에는 LOG-END-OFFSET(Producer offset), CURRENT-OFFSET(Consumer offset)이 있다.<br>
LOG-END-OFFSET에는 Producer에서 마지막으로 발행한 메시지의 offset이 저장되고,<br>
CURRENT-OFFSET은 Consumer Group에서의 offset이 저장된다.<br>

offset값을 기억하고 있으면, 예외가 발생해 프로세스가 중단되더라도, 어디서부터 다시 데이터를 처리하면 되는지 알 수 있다.<br>

또한 kakfa는,
LOG-END_OFFSET에서 CURRENT-OFFSET 을 뺀 값을 LAG로 저장한다.<br>
LAG이 계속 쌓이기만 한다면, PRODUCER가 메시지를 생성하는 속도가 CONSUMER가 처리하는 속도보다 빠르거나, CONSUMER가 정상 작동하지 않고 있다는 말이다.<br>







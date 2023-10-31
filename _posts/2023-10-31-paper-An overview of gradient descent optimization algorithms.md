---
layout: post
title: An overview of gradient descent optimization algorithms(작성중)
subtitle: 경사하강법 최적화 알고리즘 
author: san9hyun
categories: paper
banner : /assets/images/banners/book.jpg
tags: DataScience gradient optimizer 
---

# Introduction

## 논문의 목적
- 다양한 경사하강법 최적화 알고리즘들을 직관적으로 이해

## 논문 내용
- 경사하강법 최적화를 다루는 다양한 알고리즘 소개
- 각 알고리즘별 문제점/다른 알고리즘 등장 배경 소개

# Gradient descent variants

## Batch gradient descent 
모든 데이터를 사용해 비용함수에 대한 파라미터의 기울기를 계산한다.<br>

- θ = θ − η · ∇θJ(θ)

### 문제점/한계

- 메모리에 모든 데이터를 올려야함. 
- 새로운 데이터를 실시간으로 학습하는 것이 불가능.(온라인 학습 불가능)
- 한번의 업데이트가 매우 느리다.
- 데이터셋에 유사한 데이터가 있을경우 중복 계산 수행. (별로 문제처럼 안보임) 

## Stochastic gradient descent

각 샘플 $ x^{(i)} $, $ y^{(i)} $ 을 사용해 파라미터 업데이트 수행. <br>
매 에포크마다 훈련 데이터를 섞음. 

- θ = θ − η · ∇θJ(θ; $ x^{(i)} $ ; $ y^{(i)}$ }

### 장점

- Batch gradient descent의 중복 계산 문제 해결)
- 일반적으로 Batch gradient보다 빠름
- 온라인 학습 가능

### 문제점/한계

- 목적함수 수렴이 오래 걸릴 수 있음(목적 함수의 변동이 심함, 아래 그림 참고)
- 벡터 연산의 장점을 활용 못함

![SGD fluctuation](/assets/images/contents/paper/gradient descent optimizer/SGD.PNG)

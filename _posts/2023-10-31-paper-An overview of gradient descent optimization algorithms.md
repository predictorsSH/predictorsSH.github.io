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
- 새로운 데이터를 실시간으로 학습하는 것이 불가능.
- 한번의 업데이트가 매우 느리다.

## Stochastic gradient descent
$ x^{(i)} y^{(i)} $  

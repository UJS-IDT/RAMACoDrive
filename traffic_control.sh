#!/bin/bash

sudo tc qdisc del dev enp3s0 root

sudo tc qdisc add dev enp3s0 root handle 1: htb default 30

#sudo tc class add dev enp3s0 parent 1: classid 1:1 htb rate 1000Mbps

# CAV agent traffic
sudo tc class add dev enp3s0 parent 1:1 classid 1:10 htb rate 27Mbit
sudo tc class add dev enp3s0 parent 1:1 classid 1:11 htb rate 27Mbit
sudo tc class add dev enp3s0 parent 1:1 classid 1:12 htb rate 27Mbit
sudo tc class add dev enp3s0 parent 1:1 classid 1:13 htb rate 27Mbit
sudo tc class add dev enp3s0 parent 1:1 classid 1:14 htb rate 27Mbit
sudo tc class add dev enp3s0 parent 1:1 classid 1:15 htb rate 27Mbit

# Road agent traffic
sudo tc class add dev enp3s0 parent 1:1 classid 1:20 htb rate 27Mbit
sudo tc class add dev enp3s0 parent 1:1 classid 1:21 htb rate 27Mbit
sudo tc class add dev enp3s0 parent 1:1 classid 1:22 htb rate 27Mbit
sudo tc class add dev enp3s0 parent 1:1 classid 1:23 htb rate 27Mbit

# CAV agent
sudo tc filter add dev enp3s0 protocol ip parent 1:0 prio 1 u32 match ip sport 7901 0xffff flowid 1:10
sudo tc filter add dev enp3s0 protocol ip parent 1:0 prio 1 u32 match ip sport 7911 0xffff flowid 1:11
sudo tc filter add dev enp3s0 protocol ip parent 1:0 prio 1 u32 match ip sport 7921 0xffff flowid 1:12
sudo tc filter add dev enp3s0 protocol ip parent 1:0 prio 1 u32 match ip sport 7931 0xffff flowid 1:13
sudo tc filter add dev enp3s0 protocol ip parent 1:0 prio 1 u32 match ip sport 7941 0xffff flowid 1:14
sudo tc filter add dev enp3s0 protocol ip parent 1:0 prio 1 u32 match ip sport 7951 0xffff flowid 1:15

# Road agent
sudo tc filter add dev enp3s0 protocol ip parent 1:0 prio 1 u32 match ip sport 7902 0xffff flowid 1:20
sudo tc filter add dev enp3s0 protocol ip parent 1:0 prio 1 u32 match ip sport 7912 0xffff flowid 1:21
sudo tc filter add dev enp3s0 protocol ip parent 1:0 prio 1 u32 match ip sport 7922 0xffff flowid 1:22
sudo tc filter add dev enp3s0 protocol ip parent 1:0 prio 1 u32 match ip sport 7932 0xffff flowid 1:23

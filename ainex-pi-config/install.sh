#!/bin/bash
cp *.desktop ~/Desktop/
sudo cp *.service /etc/systemd/system/
sudo systemctl daemon-reload
sudo systemctl enable start_node.service expand_rootfs.service
sudo systemctl start start_node.service expand_rootfs.service 

# Makefile for data gereration
# created by Bryan Lai from HKUST

HOME_PATH = $(shell pwd)

SRC_DIR := $(HOME_PATH)/src
INC_DIR := $(HOME_PATH)/inc
BIN_DIR := $(HOME_PATH)/bin
OBJ_DIR := $(HOME_PATH)/obj

CC := g++
CCFLAGS := -g
INCLUDE := -I$(INC_DIR) \
		   -I/usr/local/cuda/include
LINK := 

SRC :=  $(filter-out ._%,$(notdir $(shell find src -name *.cpp)))
OBJ := $(SRC:%.cpp=$(OBJ_DIR)/%.o)

PROGRAM := executor

vpath %.cpp $(SRC_DIR)
vpath %.o 	$(OBJ_DIR)

all : $(PROGRAM)
	
$(PROGRAM): $(OBJ)
	$(CC) -o $(BIN_DIR)/$@ $^ $(LINK)
$(OBJ): $(OBJ_DIR)/%.o : %.cpp
	@mkdir -p $(OBJ_DIR)
	$(CC) $(CCFLAGS) -c $< -o $@ $(INCLUDE)

.PHONY: clean
clean:
	rm -rf $(OBJ_DIR)/*.o $(BIN_DIR)/$(PROGRAM)
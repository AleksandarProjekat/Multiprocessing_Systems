EXEC = parallel_tree_search

CC = gcc

CFLAGS = -fopenmp

SRC = parallel_tree_search.c

all: $(EXEC)

$(EXEC): $(SRC)
	$(CC) $(CFLAGS) -o $@ $<

run: $(EXEC)
	./$(EXEC)

clean:
	rm -f $(EXEC)


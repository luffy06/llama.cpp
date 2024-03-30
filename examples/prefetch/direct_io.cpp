#define _GNU_SOURCE
#include <stdio.h>
#include <fcntl.h>
#include <unistd.h>
#include <stdlib.h>
#include <string.h>

#define BUFFER_SIZE 4096

int main(int argc, char *argv[]) {
    int fd;
    char *buffer;
    int useDirectIO = 0; // Set to 1 to use direct I/O

    // Allocate a buffer that is 512-byte aligned
    if (posix_memalign((void **)&buffer, BUFFER_SIZE, BUFFER_SIZE) != 0) {
        perror("Error allocating aligned buffer");
        return 1;
    }

    if (argc > 2) {
        // Check if the second argument is set to use direct I/O
        if (argv[2][0] == '1') {
            useDirectIO = 1;
        }
    }

    if (argc > 1) {
        // Use the provided file name as the argument
        if (useDirectIO) {
            fd = open(argv[1], O_RDONLY | O_DIRECT);
        } else {
            fd = open(argv[1], O_RDONLY);
        }
    } else {
        // Use the default file name "test.txt"
        if (useDirectIO) {
            fd = open("ggml-model-llama-7b-q4_0-512.gguf", O_RDONLY | O_DIRECT);
        } else {
            fd = open("ggml-model-llama-7b-q4_0-512.gguf", O_RDONLY);
        }
    }

    if (fd == -1) {
        perror("Error opening file");
        free(buffer);
        return 1;
    }

    // Read data from the file in blocks until the end of file
    ssize_t bytesRead;
    do {
        bytesRead = read(fd, buffer, BUFFER_SIZE);
        if (bytesRead == -1) {
            perror("Error reading file");
            close(fd);
            free(buffer);
            return 1;
        }

        // Print the data read from the file
        //printf("Data read: %.*s\n", (int)bytesRead, buffer);
    } while (bytesRead > 0);

    sleep(100);
    // Close the file
    close(fd);

    // Free the buffer
    free(buffer);

    return 0;
}
#include <stdio.h>

int main(void)
{
    int buf = 100;
    FILE *file;
    file = fopen("test.dat", "wb");
    fwrite(&buf, sizeof(buf), 1, file);
    fclose(file);
    return 0;
}
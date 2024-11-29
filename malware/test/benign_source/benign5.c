#include <stdio.h>

int main(void)
{
    int buf;
    FILE *file;

    file = fopen("test.dat", "rb");
    fread(&buf, sizeof(buf), 1, file);
    fclose(file);

    printf("%d\n", buf);

    return 0;
}
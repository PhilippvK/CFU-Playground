#include "dev_cfu_tests.h"

#include <stdio.h>
#include <stdint.h>
#include <stdlib.h>
#include <string.h>
#include <inttypes.h>
#include "base.h"
#include "cfu.h"
#include "menu.h"
#include "riscv.h"

// Funct7 values
#define CFU_FUNCT7_SET_CODEBOOK_2    0x20
#define CFU_FUNCT7_SET_CODEBOOK_4    0x28
#define CFU_FUNCT7_SET_CODEBOOK_16   0x38
#define CFU_FUNCT7_PUSH_WEIGHTS      0x10
#define CFU_FUNCT7_ALU_MAC           0x40
#define CFU_FUNCT7_ALU_RST           0x48
#define CFU_FUNCT7_MAC_READ          0x50

static uint32_t pack_weights_2(uint8_t idx0, uint8_t idx1, uint8_t idx2, uint8_t idx3,
                               uint8_t idx4, uint8_t idx5, uint8_t idx6, uint8_t idx7) {
    return ((idx7 & 1) << 7) | ((idx6 & 1) << 6) | ((idx5 & 1) << 5) | ((idx4 & 1) << 4) |
           ((idx3 & 1) << 3) | ((idx2 & 1) << 2) | ((idx1 & 1) << 1) | ((idx0 & 1) << 0);
}
static uint32_t pack_weights_4(uint8_t idx0, uint8_t idx1, uint8_t idx2, uint8_t idx3,
                               uint8_t idx4, uint8_t idx5, uint8_t idx6, uint8_t idx7) {
    return ((idx7 & 3) << 14) | ((idx6 & 3) << 12) | ((idx5 & 3) << 10) | ((idx4 & 3) << 8) |
           ((idx3 & 3) << 6) | ((idx2 & 3) << 4) | ((idx1 & 3) << 2) | ((idx0 & 3) << 0);
}
static uint32_t pack_weights_16(uint8_t idx0, uint8_t idx1, uint8_t idx2, uint8_t idx3,
                                uint8_t idx4, uint8_t idx5, uint8_t idx6, uint8_t idx7) {
    return ((idx7 & 0xF) << 28) | ((idx6 & 0xF) << 24) | ((idx5 & 0xF) << 20) | ((idx4 & 0xF) << 16) |
           ((idx3 & 0xF) << 12) | ((idx2 & 0xF) << 8) | ((idx1 & 0xF) << 4) | ((idx0 & 0xF) << 0);
}
static uint32_t pack_activations(uint8_t a0, uint8_t a1, uint8_t a2, uint8_t a3) {
    return ((a3 & 0xFF) << 24) | ((a2 & 0xFF) << 16) | ((a1 & 0xFF) << 8) | ((a0 & 0xFF) << 0);
}

// -------- TEST CASES (now use MAC_READ after MAC op!) ------------

static void run_student_cfu_4_test(void) {
    puts("\n=== STUDENT CFU functional test (CLUSTER_NUM=4, REAL MAC, MAC_READ) ===");
    // Set clusters: 10, -21, 21, 15
    cfu_op0_hw(CFU_FUNCT7_SET_CODEBOOK_4, 0x0F15EB0A, 0);
    // Weights: indices [0,1,2,3,0,1,2,3]
    uint32_t weight_word = pack_weights_4(0,1,2,3,0,1,2,3);
    cfu_op0_hw(CFU_FUNCT7_PUSH_WEIGHTS, weight_word, 0);
    // Activations: [1,2,3,4,5,6,7,8]
    uint32_t a_lo = pack_activations(1,2,3,4);
    uint32_t a_hi = pack_activations(5,6,7,8);

    // Issue MAC op (accumulates)
    cfu_op0_hw(CFU_FUNCT7_ALU_MAC, a_lo, a_hi);
    // Issue MAC_READ to get result!
    int32_t acc = cfu_op0_hw(CFU_FUNCT7_MAC_READ, 0, 0);

    int expected = 10*1 + (-21)*2 + 21*3 + 15*4 + 10*5 + (-21)*6 + 21*7 + 15*8;
    printf("MAC (1..8) acc = %ld\n", (long)acc);
    printf("Expected acc = %d\n", expected);

    cfu_op0_hw(CFU_FUNCT7_ALU_RST, 0, 0);
}

static void run_student_cfu_2_test(void) {
    puts("\n=== STUDENT CFU functional test (CLUSTER_NUM=2, REAL MAC, MAC_READ) ===");
    // Set clusters: -43, 45
    cfu_op0_hw(CFU_FUNCT7_SET_CODEBOOK_2, 0x2DD5, 0);
    // Weights: indices [1,0,1,0,1,0,1,0]
    uint32_t weight_word = pack_weights_2(1,0,1,0,1,0,1,0);
    cfu_op0_hw(CFU_FUNCT7_PUSH_WEIGHTS, weight_word, 0);
    // Activations: [2, 3, 4, 5, 6, 7, 8, 9]
    uint32_t a_lo = pack_activations(2, 3, 4, 5);
    uint32_t a_hi = pack_activations(6, 7, 8, 9);
    // MAC, then read
    cfu_op0_hw(CFU_FUNCT7_ALU_MAC, a_lo, a_hi);
    int32_t acc = cfu_op0_hw(CFU_FUNCT7_MAC_READ, 0, 0);

    int expected = 45*2 + -43*3 + 45*4 + -43*5 + 45*6 + -43*7 + 45*8 + -43*9;
    printf("MAC (2..9) acc = %ld\n", (long)acc);
    printf("Expected acc = %d\n", expected);

    cfu_op0_hw(CFU_FUNCT7_ALU_RST, 0, 0);
}

static void run_student_cfu_16_test(void) {
    puts("\n=== STUDENT CFU functional test (CLUSTER_NUM=16, REAL MAC, MAC_READ) ===");
    // Set clusters: 0..15
    cfu_op0_hw(CFU_FUNCT7_SET_CODEBOOK_16, 0x03020100, 0x07060504);
    cfu_op0_hw(CFU_FUNCT7_SET_CODEBOOK_16, 0x0B0A0908, 0x0F0E0D0C);
    // Weights: [8,9,10,11,12,13,14,15]
    uint32_t weight_word = pack_weights_16(8,9,10,11,12,13,14,15);
    cfu_op0_hw(CFU_FUNCT7_PUSH_WEIGHTS, weight_word, 0);
    // Activations: [1,2,3,4,5,6,7,8]
    uint32_t a_lo = pack_activations(1,2,3,4);
    uint32_t a_hi = pack_activations(5,6,7,8);

    cfu_op0_hw(CFU_FUNCT7_ALU_MAC, a_lo, a_hi);
    int32_t acc = cfu_op0_hw(CFU_FUNCT7_MAC_READ, 0, 0);

    int expected = 8*1 + 9*2 + 10*3 + 11*4 + 12*5 + 13*6 + 14*7 + 15*8;
    printf("MAC (16 clusters) acc = %ld\n", (long)acc);
    printf("Expected acc = %d\n", expected);

    cfu_op0_hw(CFU_FUNCT7_ALU_RST, 0, 0);
}

struct Menu DEV_CFU_MENU = {
    "student_cfu Functional Tests",
    "student_cfu",
    {
        MENU_ITEM('2', "Run STUDENT CLUSTER_NUM=2 MAC test", run_student_cfu_2_test),
        MENU_ITEM('4', "Run STUDENT CLUSTER_NUM=4 MAC test", run_student_cfu_4_test),
        MENU_ITEM('6', "Run STUDENT CLUSTER_NUM=16 MAC test", run_student_cfu_16_test),
        MENU_END,
    },
};

void do_dev_cfu_tests(void) { menu_run(&DEV_CFU_MENU); }


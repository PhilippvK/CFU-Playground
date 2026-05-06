module tb_cfu;

    reg clk = 0;
    reg reset = 1;

    reg         cmd_valid = 0;
    wire        cmd_ready;
    reg  [9:0]  cmd_payload_function_id = 0;
    reg  [31:0] cmd_payload_inputs_0 = 0;
    reg  [31:0] cmd_payload_inputs_1 = 0;
    wire        rsp_valid;
    reg         rsp_ready = 0;
    wire [31:0] rsp_payload_outputs_0;

    // Packing helpers
    function [31:0] pack_weights_4(
        input [1:0] idx0, input [1:0] idx1, input [1:0] idx2, input [1:0] idx3,
        input [1:0] idx4, input [1:0] idx5, input [1:0] idx6, input [1:0] idx7
    );
        pack_weights_4 = (idx7 << 14) | (idx6 << 12) | (idx5 << 10) | (idx4 << 8) |
                         (idx3 << 6)  | (idx2 << 4)  | (idx1 << 2)  | (idx0 << 0);
    endfunction

    function [31:0] pack_activations(
        input [7:0] a0, input [7:0] a1, input [7:0] a2, input [7:0] a3
    );
        pack_activations = (a3 << 24) | (a2 << 16) | (a1 << 8) | (a0 << 0);
    endfunction

    // Instantiate DUT (using your "Cfu" spelling)
    Cfu dut (
        .cmd_valid(cmd_valid),
        .cmd_ready(cmd_ready),
        .cmd_payload_function_id(cmd_payload_function_id),
        .cmd_payload_inputs_0(cmd_payload_inputs_0),
        .cmd_payload_inputs_1(cmd_payload_inputs_1),
        .rsp_valid(rsp_valid),
        .rsp_ready(rsp_ready),
        .rsp_payload_outputs_0(rsp_payload_outputs_0),
        .reset(reset),
        .clk(clk)
    );

    always #5 clk = ~clk;

    initial begin
        $dumpfile("cfu_tb.vcd");
        $dumpvars(0, tb_cfu);

        // Reset phase
        reset = 1;
        cmd_valid = 0;
        rsp_ready = 0;
        cmd_payload_function_id = 0;
        cmd_payload_inputs_0 = 0;
        cmd_payload_inputs_1 = 0;
        #25; // hold reset for 25ns
        reset = 0;
        #10;

        // ---- 1. SET_CODEBOOK_4 ----
        cmd_payload_function_id = (7'h28 << 3); // funct7=0x28 on bits [9:3]
        cmd_payload_inputs_0    = 32'h0F15EB0A; // 10, -21, 21, 15
        cmd_payload_inputs_1    = 0;
        cmd_valid = 1;
        $display("TB: SET_CODEBOOK_4 funct7=0x%0h @%0t", cmd_payload_function_id[9:3], $time);
        wait (cmd_ready);
        #10;
        cmd_valid = 0;
        rsp_ready = 1;
        wait (rsp_valid);
        $display("TB: Codebook4 rsp: 0x%08h @%0t", rsp_payload_outputs_0, $time);
        rsp_ready = 0;
        #10;

        // ---- 2. PUSH_WEIGHTS ----
        cmd_payload_function_id = (7'h10 << 3); // funct7=0x10 on bits [9:3]
        cmd_payload_inputs_0 = pack_weights_4(0,1,2,3,0,1,2,3);
        cmd_payload_inputs_1 = 0;
        cmd_valid = 1;
        $display("TB: PUSH_WEIGHTS funct7=0x%0h @%0t", cmd_payload_function_id[9:3], $time);
        wait (cmd_ready);
        #10;
        cmd_valid = 0;
        rsp_ready = 1;
        wait (rsp_valid);
        $display("TB: PUSH_WEIGHTS rsp: 0x%08h @%0t", rsp_payload_outputs_0, $time);
        rsp_ready = 0;
        #10;

        // ---- 3. ALU_MAC ----
        cmd_payload_function_id = (7'h40 << 3); // funct7=0x40 on bits [9:3]
        cmd_payload_inputs_0 = pack_activations(1,2,3,4);
        cmd_payload_inputs_1 = pack_activations(5,6,7,8);
        cmd_valid = 1;
        $display("TB: ALU_MAC funct7=0x%0h @%0t", cmd_payload_function_id[9:3], $time);
        wait (cmd_ready);
        #10;
        cmd_valid = 0;
        rsp_ready = 1;
        wait (rsp_valid);
        $display("TB: ALU_MAC rsp: 0x%08h @%0t", rsp_payload_outputs_0, $time);
        rsp_ready = 0;
        #10;

        // ---- 4. MAC_READ ----
        cmd_payload_function_id = (7'h50 << 3); // funct7=0x50 on bits [9:3]
        cmd_payload_inputs_0 = 0;
        cmd_payload_inputs_1 = 0;
        cmd_valid = 1;
        $display("TB: MAC_READ funct7=0x%0h @%0t", cmd_payload_function_id[9:3], $time);
        wait (cmd_ready);
        #10;
        cmd_valid = 0;
        rsp_ready = 1;
        wait (rsp_valid);
        $display("MAC result: %0d @%0t", rsp_payload_outputs_0, $time);
        rsp_ready = 0;
        #10;

        // ---- 5. ALU_RST ----
        cmd_payload_function_id = (7'h48 << 3); // funct7=0x48 on bits [9:3]
        cmd_payload_inputs_0 = 0;
        cmd_payload_inputs_1 = 0;
        cmd_valid = 1;
        $display("TB: ALU_RST funct7=0x%0h @%0t", cmd_payload_function_id[9:3], $time);
        wait (cmd_ready);
        #10;
        cmd_valid = 0;
        rsp_ready = 1;
        wait (rsp_valid);
        $display("TB: ALU_RST rsp: 0x%08h @%0t", rsp_payload_outputs_0, $time);
        rsp_ready = 0;
        #10;

        $display("Testbench finished.");
        $finish;
    end

endmodule


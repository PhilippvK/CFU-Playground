module Cfu (
    input wire         cmd_valid,
    output wire        cmd_ready,
    input wire [9:0]   cmd_payload_function_id,
    input wire [31:0]  cmd_payload_inputs_0,
    input wire [31:0]  cmd_payload_inputs_1,
    output reg         rsp_valid,
    input              rsp_ready,
    output reg [31:0]  rsp_payload_outputs_0,
    input              reset,
    input              clk
);
    parameter PARALLEL_MULS = 8; // 1, 2, 4 or 8
    parameter MUL_STAGES = 8 / PARALLEL_MULS;
    parameter NUM_WEIGHT_BANKS = 2; // 1, 2 or 4
    // TODO: assert valid values
    reg [2:0] mac_step[NUM_WEIGHT_BANKS-1:0];   // up to 8 steps
    reg [NUM_WEIGHT_BANKS-1:0] busy;
    reg signed [31:0] acc;

    reg signed [7:0] clusters_2 [0:1][0:NUM_WEIGHT_BANKS-1];
    reg signed [7:0] clusters_4 [0:3][0:NUM_WEIGHT_BANKS-1];
    reg signed [7:0] clusters_16 [0:15][0:NUM_WEIGHT_BANKS-1];
    reg signed [7:0] active_clusters [0:31][0:NUM_WEIGHT_BANKS-1];
    integer i, j, index, bank_;
    reg signed [7:0] acts [0:7];
    reg signed [7:0] ws   [0:7];
    reg signed [15:0] prod [0:7][0:NUM_WEIGHT_BANKS-1];
    reg [1:0] active_cluster_chunk_idx[0:NUM_WEIGHT_BANKS-1];
    reg [1:0] alu_mac_count[0:NUM_WEIGHT_BANKS-1];
    reg [1:0] mac16_count[0:NUM_WEIGHT_BANKS-1];
    reg [31:0] weight_code_packed;
    reg [4:0]  last_cluster_num;
    reg        c16_toggle;
    reg signed [31:0] total_mac;
    reg signed [31:0] sum [0:7][0:NUM_WEIGHT_BANKS-1];

    wire [6:0] funct7 = cmd_payload_function_id[9:3];
    wire [2:0] funct3 = cmd_payload_function_id[2:0];
    // funct3[1:0]: bank idx (0,1,2,3)
    wire [$clog2(NUM_WEIGHT_BANKS)-1:0] bank = funct3[$clog2(NUM_WEIGHT_BANKS)-1:0];
    // funct3[2]: broadcast flag (0,1)
    wire broadcast = funct3[2];
    localparam [6:0] CFU_FUNCT7_SET_CODEBOOK_2   = 7'h20;
    localparam [6:0] CFU_FUNCT7_SET_CODEBOOK_4   = 7'h28;
    localparam [6:0] CFU_FUNCT7_SET_CODEBOOK_16  = 7'h38;
    localparam [6:0] CFU_FUNCT7_PUSH_WEIGHTS     = 7'h10;
    localparam [6:0] CFU_FUNCT7_ALU_MAC          = 7'h40;
    localparam [6:0] CFU_FUNCT7_ALU_RST          = 7'h48;
    localparam [6:0] CFU_FUNCT7_MAC_READ         = 7'h50;
    localparam [6:0] CFU_FUNCT7_DEBUG_DUMP       = 7'h52;
    localparam [6:0] CFU_FUNCT7_MAC_READ_NO_RESET = 7'h54;

    assign cmd_ready = (!rsp_valid); // Do not accept new commands until response is accepted

    always @(posedge clk) begin
        if (reset) begin
            for (i = 0; i < NUM_WEIGHT_BANKS; i = i + 1) begin
                active_cluster_chunk_idx[i] <= 0;
                alu_mac_count[i] <= 0;
                mac16_count[i] <= 0;
            end
            total_mac <= 0;
            for (i = 0; i < 2; i = i + 1) begin
                for (j = 0; j < NUM_WEIGHT_BANKS; j = j + 1) begin
                    clusters_2[i][j] <= 0;
                end
            end
            for (i = 0; i < 4; i = i + 1) begin
                for (j = 0; j < NUM_WEIGHT_BANKS; j = j + 1) begin
                    clusters_4[i][j] <= 0;
                end
            end
            for (i = 0; i < 16; i = i + 1) begin
                for (j = 0; j < NUM_WEIGHT_BANKS; j = j + 1) begin
                    clusters_16[i][j] <= 0;
                end
            end
            for (i = 0; i < 32; i = i + 1) begin
                for (j = 0; j < NUM_WEIGHT_BANKS; j = j + 1) begin
                    active_clusters[i][j] <= 0;
                end
            end
            weight_code_packed <= 0;
            last_cluster_num <= 4;
            c16_toggle <= 0;
            for (i = 0; i < 8; i = i + 1) begin
                for (j = 0; j < NUM_WEIGHT_BANKS; j = j + 1) begin
                    sum[i][j] <= 0;
                end
            end
            for (j = 0; j < NUM_WEIGHT_BANKS; j = j + 1) begin
                mac_step[j] <= 0;
            end
            for (j = 0; j < NUM_WEIGHT_BANKS; j = j + 1) begin
                busy[j] <= 0;
            end
            rsp_valid <= 0;
            rsp_payload_outputs_0 <= 0;
        end
        else begin
            // Handle response handshake
            if (rsp_valid && rsp_ready)
                rsp_valid <= 0;

            // Only accept new commands when not holding a response
            if (cmd_valid && !rsp_valid && !busy && funct7 == CFU_FUNCT7_ALU_MAC) begin
                if (last_cluster_num == 16) begin
                    // For 16 clusters: use active_clusters[0..7] first, [8..15] next
                    for (j = 0; j < 4; j = j+1) begin
                        acts[j]   = $signed(cmd_payload_inputs_0 >> (8*j));
                        ws[j][bank]     = active_clusters[mac16_count[bank]*8 + j][bank];
                    end
                    for (j = 0; j < 4; j = j+1) begin
                        acts[j+4] = $signed(cmd_payload_inputs_1 >> (8*j));
                        ws[j+4][bank]   = active_clusters[mac16_count[bank]*8 + j + 4][bank];
                    end
                    // Increment mac16_count, reset after two calls
                    if (mac16_count[bank] == 1)
                        mac16_count[bank] <= 0;
                    else
                        mac16_count[bank] <= mac16_count[bank] + 1;
                end else begin
                    // For 2/4 clusters, chunk by alu_mac_count
                    for (j = 0; j < 4; j = j+1) begin
                        acts[j]   = $signed(cmd_payload_inputs_0 >> (8*j));
                        ws[j][bank]     = active_clusters[alu_mac_count[bank]*8+j][bank];
                    end
                    for (j = 0; j < 4; j = j+1) begin
                        acts[j+4] = $signed(cmd_payload_inputs_1 >> (8*j));
                        ws[j+4][bank]   = active_clusters[alu_mac_count[bank]*8+j+4][bank];
                    end
                    alu_mac_count[bank] <= alu_mac_count[bank] + 1;
                end
                if (broadcast) begin
                    for (bank_ = 0; bank_ < NUM_WEIGHT_BANKS; bank_ = bank_ + 1) begin
                        busy[bank_] <= 1'b1;
                    end
                end
                else begin
                    busy[bank] <= 1'b1;
                end
                // TODO: handle broadcast (same activations)

                // Common accumulation for both
                // for (j = 0; j < 8; j = j+1) begin
                //     prod[j] = ws[j] * (acts[j] + 128);
                // end
                // sum[0] <= sum[0] + prod[0];
                // sum[1] <= sum[1] + prod[1];
                // sum[2] <= sum[2] + prod[2];
                // sum[3] <= sum[3] + prod[3];
                // sum[4] <= sum[4] + prod[4];
                // sum[5] <= sum[5] + prod[5];
                // sum[6] <= sum[6] + prod[6];
                // sum[7] <= sum[7] + prod[7];
                // rsp_payload_outputs_0 <= 32'hABCD0001;
            end
            else if (busy > 0) begin
                // TODO: handle multiple busy bits 1 at the same time? -> return val
                for (bank_ = 0; bank_ < NUM_WEIGHT_BANKS; bank_ = bank_ + 1) begin
                    if (busy[bank_]) begin
                        for (j = 0; j < PARALLEL_MULS; j = j+1) begin
                            index = mac_step[bank_] * PARALLEL_MULS + j;
                            if (index < 8) begin
                                prod[index][bank_] = ws[index][bank_] * (acts[index] + 128);
                                sum[index][bank_] <= sum[index][bank_] + prod[index][bank_];
                            end
                        end

                        if ((mac_step[bank_] + 1) >= MUL_STAGES) begin
                            // DONE
                            // TODO: decoupled?
                            rsp_payload_outputs_0 <= 32'hABCD0001;
                            rsp_valid <= 1;
                            busy[bank_] <= 0;
                        end else begin
                            mac_step[bank_] <= mac_step[bank_] + 1;
                        end
                    end
                end
            end
            else if (cmd_valid && !rsp_valid) begin
                rsp_valid <= 1'b1;

                if (funct7 == CFU_FUNCT7_SET_CODEBOOK_2) begin
                    clusters_2[0][bank] <= $signed(cmd_payload_inputs_0[7:0]);
                    clusters_2[1][bank] <= $signed(cmd_payload_inputs_0[15:8]);
                    last_cluster_num <= 2;
                    rsp_payload_outputs_0 <= 32'hAABB2202;
                end
                else if (funct7 == CFU_FUNCT7_SET_CODEBOOK_4) begin
                    clusters_4[0][bank] <= $signed(cmd_payload_inputs_0[7:0]);
                    clusters_4[1][bank] <= $signed(cmd_payload_inputs_0[15:8]);
                    clusters_4[2][bank] <= $signed(cmd_payload_inputs_0[23:16]);
                    clusters_4[3][bank] <= $signed(cmd_payload_inputs_0[31:24]);
                    last_cluster_num <= 4;
                    rsp_payload_outputs_0 <= 32'hAABB4404;
                end
                else if (funct7 == CFU_FUNCT7_SET_CODEBOOK_16) begin
                    if (!c16_toggle) begin
                        clusters_16[0][bank] <= $signed(cmd_payload_inputs_0[7:0]);
                        clusters_16[1][bank] <= $signed(cmd_payload_inputs_0[15:8]);
                        clusters_16[2][bank] <= $signed(cmd_payload_inputs_0[23:16]);
                        clusters_16[3][bank] <= $signed(cmd_payload_inputs_0[31:24]);
                        clusters_16[4][bank] <= $signed(cmd_payload_inputs_1[7:0]);
                        clusters_16[5][bank] <= $signed(cmd_payload_inputs_1[15:8]);
                        clusters_16[6][bank] <= $signed(cmd_payload_inputs_1[23:16]);
                        clusters_16[7][bank] <= $signed(cmd_payload_inputs_1[31:24]);
                        rsp_payload_outputs_0 <= 32'hAABB16A0;
                    end else begin
                        clusters_16[8][bank] <= $signed(cmd_payload_inputs_0[7:0]);
                        clusters_16[9][bank] <= $signed(cmd_payload_inputs_0[15:8]);
                        clusters_16[10][bank] <= $signed(cmd_payload_inputs_0[23:16]);
                        clusters_16[11][bank] <= $signed(cmd_payload_inputs_0[31:24]);
                        clusters_16[12][bank] <= $signed(cmd_payload_inputs_1[7:0]);
                        clusters_16[13][bank] <= $signed(cmd_payload_inputs_1[15:8]);
                        clusters_16[14][bank] <= $signed(cmd_payload_inputs_1[23:16]);
                        clusters_16[15][bank] <= $signed(cmd_payload_inputs_1[31:24]);
                        rsp_payload_outputs_0 <= 32'hAABB16B1;
                    end
                    c16_toggle <= ~c16_toggle;
                    last_cluster_num <= 16;
                end
                else if (funct7 == CFU_FUNCT7_PUSH_WEIGHTS) begin
                    weight_code_packed <= cmd_payload_inputs_0;
                    if (last_cluster_num == 2) begin
                        for (i = 0; i < 32; i = i + 1)
                            active_clusters[i][bank] <= clusters_2[(cmd_payload_inputs_0 >> i) & 1][bank];
                    end
                    else if (last_cluster_num == 4) begin
                        for (i = 0; i < 16; i = i + 1)
                            active_clusters[i][bank] <= clusters_4[cmd_payload_inputs_0[(2*i)+1 -: 2]][bank];
                        for (i = 0; i < 16; i = i + 1)
                            active_clusters[16 + i][bank] <= clusters_4[cmd_payload_inputs_1[(2*i)+1 -: 2]][bank];
                    end
                    else if (last_cluster_num == 16) begin
                        for (i = 0; i < 8; i = i + 1) begin
                            active_clusters[i][bank] <= clusters_16[cmd_payload_inputs_0[i*4 +: 4]][bank];
                            active_clusters[8 + i][bank] <= clusters_16[cmd_payload_inputs_1[i*4 +: 4]][bank];
                        end
                    end

                    rsp_payload_outputs_0 <= 32'hDEAD0000;
                    active_cluster_chunk_idx[bank] <= active_cluster_chunk_idx[bank] + 1;
                end
                /*
                else if (funct7 == CFU_FUNCT7_ALU_MAC) begin
                    for (j = 0; j < 4; j = j+1) begin
                        acts[j]   = $signed(cmd_payload_inputs_0 >> (8*j));
                        ws[j]     = active_clusters[alu_mac_count*8+j];
                    end
                    for (j = 0; j < 4; j = j+1) begin
                        acts[j+4] = $signed(cmd_payload_inputs_1 >> (8*j));
                        ws[j+4]   = active_clusters[alu_mac_count*8+j+4];
                    end
                    for (j = 0; j < 8; j = j+1) begin
                        prod[j] = ws[j] * (acts[j] + 128);
                    end
                    sum[0] <= sum[0] + prod[0];
                    sum[1] <= sum[1] + prod[1];
                    sum[2] <= sum[2] + prod[2];
                    sum[3] <= sum[3] + prod[3];
                    sum[4] <= sum[4] + prod[4];
                    sum[5] <= sum[5] + prod[5];
                    sum[6] <= sum[6] + prod[6];
                    sum[7] <= sum[7] + prod[7];
                    alu_mac_count <= alu_mac_count + 1;
                    rsp_payload_outputs_0 <= 32'hABCD0001;
                end
                */
                else if (funct7 == CFU_FUNCT7_MAC_READ) begin
                    rsp_payload_outputs_0 <= sum[0][bank] + sum[1][bank] + sum[2][bank] + sum[3][bank] +
                       sum[4][bank] + sum[5][bank] + sum[6][bank] + sum[7][bank];
                end
                else if (funct7 == CFU_FUNCT7_MAC_READ_NO_RESET) begin
                    rsp_payload_outputs_0 <= sum[0][bank] + sum[1][bank] + sum[2][bank] + sum[3][bank] +
                       sum[4][bank] + sum[5][bank] + sum[6][bank] + sum[7][bank];
                end
                else if (funct7 == CFU_FUNCT7_ALU_RST) begin
                    total_mac <= 0;
                    for (i = 0; i < 8; i = i + 1) begin
                        sum[i][bank] <= 0;
                    end
                    alu_mac_count[bank] <= 0;
                    mac16_count[bank] <= 0;
                    active_cluster_chunk_idx[bank] <= 0;
                    rsp_payload_outputs_0 <= 0;
                end
                else if (funct7 == CFU_FUNCT7_DEBUG_DUMP) begin
                    rsp_payload_outputs_0 <= active_clusters[cmd_payload_inputs_0[4:0]][bank];
                end
                else begin
                    rsp_payload_outputs_0 <= 0;
                end
            end

            // Clear sums and indices only AFTER MAC_READ is acknowledged
            if (rsp_valid && rsp_ready && funct7 == CFU_FUNCT7_MAC_READ) begin
                for (i = 0; i < 8; i = i + 1) begin
                    sum[i][bank] <= 0;
                end
                alu_mac_count[bank] <= 0;
                mac16_count[bank] <= 0;
                active_cluster_chunk_idx[bank] <= 0;
            end
        end
    end

endmodule

/*
 * Copyright 2020-2022 NXP
 * All rights reserved.
 *
 * SPDX-License-Identifier: BSD-3-Clause
 */

#include "fsl_debug_console.h"
#include "output_postproc.h"
#include "get_top_n.h"
#include "demo_config.h"
#include "labels.h"

#ifdef EIQ_GUI_PRINTF
#include "chgui.h"
#endif

status_t MODEL_ProcessOutput(const uint8_t* data,
                             const tensor_dims_t* dims,
                             tensor_type_t type,
                             int inferenceTime)
{
    /* Detection threshold */
    const float threshold = (float)DETECTION_TRESHOLD / 100;

    /* Store top results */
    result_t topResults[NUM_RESULTS];

    /* Get top prediction */
    MODEL_GetTopN(data,
                  dims->data[dims->size - 1],
                  type,
                  NUM_RESULTS,
                  threshold,
                  topResults);

    /* ==============================
       Variables
       ============================== */

    float detected_score = 0.0f;
    float other_score = 0.0f;

    int detected_index = -1;

    const char* detected_label = "Unknown";

    /* ==============================
       Get detected class
       ============================== */

    if (topResults[0].index >= 0)
    {
        detected_index = topResults[0].index;

        /* Convert score to percentage */
        detected_score = topResults[0].score * 100.0f;

        /* Remaining percentage */
        other_score = 100.0f - detected_score;

        detected_label = labels[detected_index];
    }

    /* ==============================
       Print Output
       ============================== */

    PRINTF("----------------------------------------" EOL);
    PRINTF("Inference time: %d us" EOL, inferenceTime);

    /*
     * Class 0 = Inorganic
     * Class 1 = Organic
     */

    if (detected_index == 0)
    {
        /* Inorganic detected */

        PRINTF("Class 0 (%s): %d%%" EOL,
               labels[0],
               (int)detected_score);

        PRINTF("Class 1 (%s): %d%%" EOL,
               labels[1],
               (int)other_score);
    }
    else if (detected_index == 1)
    {
        /* Organic detected */

        PRINTF("Class 0 (%s): %d%%" EOL,
               labels[0],
               (int)other_score);

        PRINTF("Class 1 (%s): %d%%" EOL,
               labels[1],
               (int)detected_score);
    }
    else
    {
        /* No valid detection */

        PRINTF("Class 0 (%s): 0%%" EOL,
               labels[0]);

        PRINTF("Class 1 (%s): 0%%" EOL,
               labels[1]);
    }

    PRINTF("Detected Image: %s" EOL,
           detected_label);

    PRINTF("----------------------------------------" EOL);

#ifdef EIQ_GUI_PRINTF
    GUI_PrintfToBuffer(GUI_X_POS,
                       GUI_Y_POS,
                       "Detected: %s",
                       detected_label);
#endif

    return kStatus_Success;
}

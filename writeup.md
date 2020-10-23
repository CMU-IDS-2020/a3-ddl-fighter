# Project name

![A screenshot of your application. Could be a GIF.](screenshot.pdf)


## Project Goals

In this project, we want to let the user to explore how businesses reacted to the COVID-19, and how different factors affect their reaction. We mainly focus on three factors: the businesses' location, the businesses' category, the businesses' quality.

## Design
We explore location and business reaction in Section 3. As there are only several data point in each county(which means that the visulization is very sparse), we choose to only visualize the geometric figure in the state level. Then, we assume that the user will want to futher explore why there are differences in different locations, so we use scatter plot to enable them to find some possible insight.

We explore categories and business reaction in Section 3, especially in terms of Temporary Closed Until, Covid Banner and highlights. We use color to encode categories, because 1) color is recommended for encoding norminal features and 2) we mainly want to use stacked bar charts to show a decomposition of certain reaction into categories. For interactivity, we allow readers to select target COVID features as we have done before. Meanwhile, in both section 3.1&3, we allow readers to select time window to explore in more detail. When reviewing our design, we assume readers may be interested in comparison within a category, and therefore additionally for 3.3, we support highlights bars of a selected category.

For section 4.1, we explore how quality affects interaction. Here we choose the scatter plot given that we have two quantitative variables and one norminal. We start by overviewing rating stars and review counts. We initially made it a static graph, but later we assume readers might want to see how these two factors interact, and thus support intersected interval selection. In 4.2, we choose the scatter plotallow readers to "design" their target scatter plot, with deciding a target, a sample size and a filter. Due to the rating stars are discrete, the plot could be crowded. Therefore, we additional support highlight on selection.


## Development

Haonan Wang finished section 1 first half, section 2 and section 5. The project roughly took him 20 hours, and most of his time spent on align the geometric data and join different dataset by geometric information.
Weiyi Zhang finished section1 second half, section 3 and section 4. The project roughly took her 25 hours These things took weiyi the most time: Group data into certain formation through Pandas and Numpy because altair does not quite support huge dataset. Also, adding interactive widgets for altair is tricky, with some operations leading to unexpected effects or errors.

Two students first develop a overall structure for the interactions. As they are in differnet time zone, one student worked during Pittsburgh's daytime, while the other student worked during Pittsburgh's night time. They synced every 12 hours about their progress and discussed about what should add.


USE [WyethDBA]
GO
IF  EXISTS (SELECT * FROM sys.views WHERE object_id = OBJECT_ID(N'[dbo].[vw_FailScheduledJob]'))
DROP VIEW [dbo].[vw_FailScheduledJob]
GO
SET ANSI_NULLS ON
GO
SET QUOTED_IDENTIFIER ON
GO
CREATE	View [dbo].[vw_FailScheduledJob]
As
/*********************************************************************/
/* Database: WyethDBA  											     */
/* Purpose: Any Failure View									     */
/* Note: 	This View Is used by P_Monitor_ScheduleJobFailure Proc   */
/*           to Generate EMAIL									     */
/*                                                                   */
/*  Org. Date: 09/06/2008        EPAK                                */
/*  Mod. Date: 00/00/0000                                            */
/*********************************************************************/
Select	SVR.SiteID,
		SVR.ServerID,
		ST.ServerTypeID,
		DIF.DateID,
		SVR.ServerName,
		DS.SiteDesc,
		ST.ServerTypeDesc,		
		'[' + ST.ServerTypeDesc + '] ' + SVR.ServerName +
		' Server at ' + DS.SiteDesc +
		+  ' Site - ' +
		SFC.ScheduleFreqDesc + 
		' Scheduled Job Failure [' +
		SSJ.JobDesc + 
		'] Occured on ' + 
		DIF.JobDateInfo As FailureInfo
From	DatabaseServerScheduledJob SSJ LEFT JOIN DatabaseServer SVR
ON		SSJ.ServerID = SVR.ServerID
		INNER JOIN DateInfo DIF
ON		SSJ.DateID = DIF.DateID
		INNER JOIN DatabaseScheduleFrequency SFC
ON		SSJ.ScheduleFreqCode = SFC.ScheduleFreqCode
		INNER JOIN DatabaseSite DS
ON		SVR.SiteID = DS.SiteID	
		INNER JOIN DatabaseServerType ST
ON		SVR.ServerTypeID = ST.ServerTypeID
Where	SSJ.DateID = (Select Max(DateID) From DateInfo Where ActiveFlag = 'Y')
And		SSJ.JobEnableCode = 1
And		SSJ.ScheduleEnableCode = 1
And		SSJ.JobStatusCode = 0
UNION
Select	SVR.SiteID,
		SVR.ServerID,
		ST.ServerTypeID,
		DIF.DateID,
		SVR.ServerName,
		DS.SiteDesc,
		ST.ServerTypeDesc,
		'[' + ST.ServerTypeDesc + '] ' + SVR.ServerName +
		' Server at ' + DS.SiteDesc +
		' Site - ' +
		SFC.ScheduleFreqDesc + 
		' Scheduled Job IN Progress [' +
		SSJ.JobDesc + 
		'] Occured on ' + 
		DIF.JobDateInfo + '. [Last Completion Date: ' + JobLastRunDate + ']' As FailureInfo
From	DatabaseServerScheduledJob SSJ LEFT JOIN DatabaseServer SVR
ON		SSJ.ServerID = SVR.ServerID
		INNER JOIN DateInfo DIF
ON		SSJ.DateID = DIF.DateID
		INNER JOIN DatabaseScheduleFrequency SFC
ON		SSJ.ScheduleFreqCode = SFC.ScheduleFreqCode
		INNER JOIN DatabaseSite DS
ON		SVR.SiteID = DS.SiteID	
		INNER JOIN DatabaseServerType ST
ON		SVR.ServerTypeID = ST.ServerTypeID
Where	SSJ.DateID = (Select Max(DateID) From DateInfo Where ActiveFlag = 'Y')
And		SSJ.JobEnableCode = 1
And		SSJ.ScheduleEnableCode = 1
And		SSJ.JobStatusCode = 4
And		SSJ.JobDesc NOT LIKE '%Collect Database & Storage Infomation%'
And		SSJ.JobDesc NOT LIKE '%Validate Link Server Connect & Add Server Version%'
And		DATEDIFF(hh,(CONVERT(Datetime,SSJ.JobLastRunDate + ' ' + wyethdba.dbo.fnReverseTimeFormat(SSJ.JobLastRunTime))),SSJ.LastUpdate) >= 6
/*********************************************************************/
/*                                                                   */
/*********************************************************************/
GO
GRANT  SELECT  ON [dbo].[vw_FailScheduledJob]  TO [WyethDBA_Client]